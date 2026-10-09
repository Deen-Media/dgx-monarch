"""Generate ComfyUI workflows from graph specifications.

Public templates go in example_workflows/; sweep fixtures go in
tests/fixtures/workflows/generated/ and do not appear in the template browser.

DGX Monarch widget order comes from INPUT_TYPES. Stock ComfyUI nodes use the
explicit STOCK map. Regenerate and commit the JSON after changing either:

    python tools/gen_templates.py

``--check`` writes nothing and exits 1 if generated files differ.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
import types
from collections.abc import Callable, Iterator

# The sibling src, so `python tools/gen_templates.py` runs from any directory
# and a worktree reads its own node definitions. The entry leaves sys.path
# once the package is imported: its submodules resolve through the package's
# own __path__, and a process that imports this file as a library (the sweep
# converter, the tests) keeps the sys.path it had.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_ADDED_SRC = _SRC not in sys.path
if _ADDED_SRC:
    sys.path.insert(0, _SRC)
try:
    from dgx_monarch.nodes import NODE_CLASS_MAPPINGS
finally:
    if _ADDED_SRC:
        sys.path.remove(_SRC)

# Comfy stubs: INPUT_TYPES reads dropdown lists from comfy. The stubs enter
# sys.modules only inside comfy_stubs(), for one call, never at import. Never
# install them at module level: a stub left in sys.modules made a later
# real-comfy test meet a bare `comfy` with no __path__ and fail to import
# comfy.options (2026-10-05).
fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda kind: ["stub.safetensors"]
fp.get_output_directory = lambda: "/tmp"
fp.base_path = "/tmp"
comfy = types.ModuleType("comfy")
samplers_mod = types.ModuleType("comfy.samplers")


class _KS:
    SAMPLERS = ["euler"]
    SCHEDULERS = ["simple"]


samplers_mod.KSampler = _KS
comfy.samplers = samplers_mod
STUBS = {"folder_paths": fp, "comfy": comfy, "comfy.samplers": samplers_mod}


@contextlib.contextmanager
def comfy_stubs() -> Iterator[None]:
    """Temporarily install the three ComfyUI stubs and restore previous entries.

    Replace existing entries, including incomplete stubs from other tests, so
    ``comfy.samplers`` remains importable. Remove entries that did not exist on entry.
    """
    before = {name: sys.modules[name] for name in STUBS if name in sys.modules}
    sys.modules.update(STUBS)
    try:
        yield
    finally:
        for name in STUBS:
            sys.modules.pop(name, None)
        sys.modules.update(before)


def _with_comfy_stubs(function: Callable) -> Callable:
    """Run the function under comfy_stubs(); nesting is cheap and exact."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with comfy_stubs():
            return function(*args, **kwargs)

    return wrapped


WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
# The frontend gives a seed its own control widget and stores that control in
# the next positional slot of widgets_values. It attaches one to any INT input
# that asks for it, and, when the input says nothing, to any INT named `seed`
# or `noise_seed` (comfyui_frontend_package 1.48.7,
# src/renderer/extensions/vueNodes/widgets/composables/useIntWidget.ts:
# `inputSpec.control_after_generate ?? ['seed', 'noise_seed'].includes(name)`).
# The combo is created right after its seed widget and LGraphNode.serialize
# walks node.widgets in order, so the shipped value belongs directly after the
# seed. Comfy's own default graph stores it that way:
# `[42, 'fixed', 8, 1, 'res_multistep', 'simple', 1]`. A template that leaves
# the slot out hands the control the next widget's value, which is not a
# control name, so the frontend falls back to `fixed` and a second Queue press
# re-submits the same graph and is served from the execution cache.
CONTROL_WIDGET = "control_after_generate"
CONTROL_VALUE = "randomize"
SEED_WIDGET_NAMES = ("seed", "noise_seed")
# Stock Comfy nodes list widget names in order; connection inputs are excluded.
STOCK = {
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "CLIPTextEncode": ["text"],
    "TextEncodeMageFlowEdit": ["prompt", "negative_prompt", "width", "height", "batch_size"],
    # The native Qwen Image 2.1 node has two text widgets followed by the
    # reference-image target resolution. `vae` and its autogrowing images are
    # sockets, not widgets; the generator declares all 16 image sockets below
    # so image_10 never changes index when a reader adds more.
    "TextEncodeQwenImage21": ["prompt", "negative_prompt", "resolution"],
    "QwenImage21Cache": ["device", "dtype"],
    "CLIPVisionLoader": ["clip_name"],
    "CLIPVisionEncode": ["crop"],
    "ConditioningZeroOut": [],
    "FluxGuidance": ["guidance"],
    "PiDConditioning": ["latent_format", "degrade_sigma"],
    "InstructPixToPixConditioning": [],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "EmptyMiniMaxH3LatentAV": ["width", "height", "length"],
    "MiniMaxH3ImageToVideo": ["prompt", "width", "height", "length"],
    "MiniMaxH3AddGuide": ["frame_idx"],
    # A V3 schema node: comfy converts it required-then-optional, so the one
    # optional widget, replacement_mode, lands last rather than beside
    # batch_size where the schema declares it. Cross-checked against comfy's
    # own shipped SCAIL-2 template, which serializes this node as
    # [512, 896, 65, 1, 1, 0, 1, 0, 5, true].
    "WanSCAILToVideo": ["width", "height", "length", "batch_size",
                        "pose_strength", "pose_start", "pose_end",
                        "video_frame_offset", "previous_frame_count",
                        "replacement_mode"],
    "WanImageToVideo": ["width", "height", "length", "batch_size"],
    # Wan-Animate2 is a V3 schema node. Its scalar inputs retain this order
    # when serialized to the legacy UI graph; media and conditioning are sockets.
    "WanAnimate2ToVideo": ["width", "height", "length", "batch_size",
                           "video_frame_offset", "pose_strength",
                           "pose_start_percent", "pose_end_percent",
                           "reference_image_strength"],
    "TrimVideoLatent": ["trim_amount"],
    "WanDancerEncodeAudio": ["video_frames", "audio_inject_scale"],
    "WanDancerVideo": ["width", "height", "length"],
    "WanDancerPadKeyframesList": ["segment_length", "num_segments"],
    "BerniniConditioning": ["width", "height", "length", "batch_size",
                            "ref_max_size"],
    "EmptyHunyuanLatentVideo": ["width", "height", "length", "batch_size"],
    "EmptyHunyuanVideo15Latent": ["width", "height", "length", "batch_size"],
    "EmptyHunyuanImageLatent": ["width", "height", "batch_size"],
    # noise_augmentation is required, and marked advanced, so the frontend
    # draws it like any other widget.
    "HunyuanRefinerLatent": ["noise_augmentation"],
    "EmptyLTXVLatentVideo": ["width", "height", "length", "batch_size"],
    "EmptyFlux2LatentImage": ["width", "height", "batch_size"],
    "EmptyChromaRadianceLatentImage": ["width", "height", "batch_size"],
    "CLIPTextEncodeKandinsky5": ["clip_l", "qwen25_7b"],
    "LTXVConditioning": ["frame_rate"],
    "LTXVEmptyLatentAudio": ["frames_number", "frame_rate", "batch_size"],
    "LTXVConcatAVLatent": [],
    "LTXVSeparateAVLatent": [],
    "LTXVImgToVideo": ["width", "height", "length", "batch_size", "strength"],
    "LTXVAddGuide": ["frame_idx", "strength"],
    "LTXVCropGuides": [],
    "LTXVLatentUpsampler": [],
    # image_upload puts a second widget after the filename; the official LTX
    # 2.5 templates serialize it as ["<file>.png", "image"].
    "LoadImage": ["image", "upload"],
    "LoadImageMask": ["image", "channel"],
    "JoinImageWithAlpha": [],
    # video_upload puts the same second widget after the filename combo that
    # image_upload puts after LoadImage's: the frontend's Comfy.UploadImage
    # extension appends one required input named `upload` whose stored value is
    # the literal "image" for every media kind.
    "LoadVideo": ["file", "upload"],
    # The audio upload extension appends a player widget and an upload button
    # after the filename, and the official WanDancer template serializes the
    # node as ["<file>.mp3", null, null].
    "LoadAudio": ["audio", "audioUI", "upload"],
    "GetVideoComponents": [],
    "GetImageSize": [],
    "ImageFromBatch": ["batch_index", "length"],
    "ImageScale": ["upscale_method", "width", "height", "crop"],
    "ImageToMask": ["channel"],
    "MaskToImage": [],
    "LatentCutToBatch": ["dim", "slice_size"],
    "VAEEncode": [],
    "LTXVScheduler": ["steps", "max_shift", "base_shift", "stretch", "terminal"],
    "ManualSigmas": ["sigmas"],
    "Ideogram4Scheduler": ["steps", "width", "height", "mu", "std"],
    "KSamplerSelect": ["sampler_name"],
    "RandomNoise": ["noise_seed"],
    "LatentUpscaleModelLoader": ["model_name"],
    "VAELoader": ["vae_name"],
    "VAEDecode": [],
    "VAEDecodeTiled": ["tile_size", "overlap", "temporal_size", "temporal_overlap"],
    "VAEDecodeAudio": [],
    "LTXVAudioVAEDecode": [],
    "SaveImage": ["filename_prefix"],
    "SaveWEBM": ["filename_prefix", "codec", "fps", "crf"],
    # Create Video is V3. Every declared widget retains a positional slot even
    # when a graph wires it: the converter consumes that slot while preserving
    # the link, so bit_depth/color_space/codec never shift behind a linked FPS.
    "CreateVideo": ["fps", "bit_depth", "color_space", "codec"],
    "SaveVideo": ["filename_prefix", "format", "codec"],
    "SaveAudioAdvanced": ["filename_prefix", "format"],
    "Note": [],  # Note stores its text in widgets_values directly
}
STOCK_OUTPUTS = {
    "DualCLIPLoader": [("CLIP", "CLIP")],
    "CLIPLoader": [("CLIP", "CLIP")],
    "CLIPTextEncode": [("CONDITIONING", "CONDITIONING")],
    "TextEncodeMageFlowEdit": [("positive", "CONDITIONING"),
                               ("negative", "CONDITIONING"), ("latent", "LATENT")],
    "TextEncodeQwenImage21": [("positive", "CONDITIONING"),
                                ("negative", "CONDITIONING"),
                                ("latent", "LATENT")],
    "QwenImage21Cache": [("MODEL", "MODEL")],
    "CLIPVisionLoader": [("CLIP_VISION", "CLIP_VISION")],
    "CLIPVisionEncode": [("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT")],
    "ConditioningZeroOut": [("CONDITIONING", "CONDITIONING")],
    "FluxGuidance": [("CONDITIONING", "CONDITIONING")],
    "PiDConditioning": [("CONDITIONING", "CONDITIONING")],
    "InstructPixToPixConditioning": [("positive", "CONDITIONING"),
                                     ("negative", "CONDITIONING"),
                                     ("latent", "LATENT")],
    "EmptySD3LatentImage": [("LATENT", "LATENT")],
    "EmptyLatentImage": [("LATENT", "LATENT")],
    "EmptyMiniMaxH3LatentAV": [("LATENT", "LATENT")],
    "MiniMaxH3ImageToVideo": [("positive", "CONDITIONING"), ("LATENT", "LATENT")],
    "MiniMaxH3AddGuide": [("positive", "CONDITIONING")],
    "WanSCAILToVideo": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                        ("latent", "LATENT"), ("video_frame_offset", "INT")],
    "WanImageToVideo": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                        ("latent", "LATENT")],
    "WanAnimate2ToVideo": [("positive", "CONDITIONING"),
                             ("negative", "CONDITIONING"), ("latent", "LATENT"),
                             ("trim_latent", "INT"), ("trim_image", "INT"),
                             ("video_frame_offset", "INT")],
    "TrimVideoLatent": [("LATENT", "LATENT")],
    "WanDancerEncodeAudio": [("audio_encoder_output", "AUDIO_ENCODER_OUTPUT"),
                             ("fps_string", "STRING")],
    "WanDancerVideo": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                       ("latent", "LATENT")],
    "WanDancerPadKeyframesList": [("keyframes_sequence", "IMAGE"),
                                  ("keyframes_mask", "MASK"),
                                  ("audio_segment", "AUDIO")],
    "BerniniConditioning": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                            ("latent", "LATENT")],
    "EmptyHunyuanLatentVideo": [("LATENT", "LATENT")],
    "EmptyHunyuanVideo15Latent": [("LATENT", "LATENT")],
    "EmptyHunyuanImageLatent": [("LATENT", "LATENT")],
    "HunyuanRefinerLatent": [("positive", "CONDITIONING"),
                             ("negative", "CONDITIONING"),
                             ("latent", "LATENT")],
    "EmptyLTXVLatentVideo": [("LATENT", "LATENT")],
    "EmptyFlux2LatentImage": [("LATENT", "LATENT")],
    "EmptyChromaRadianceLatentImage": [("LATENT", "LATENT")],
    "CLIPTextEncodeKandinsky5": [("CONDITIONING", "CONDITIONING")],
    "LTXVConditioning": [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
    "LTXVEmptyLatentAudio": [("Latent", "LATENT")],
    "LTXVConcatAVLatent": [("latent", "LATENT")],
    "LTXVSeparateAVLatent": [("video_latent", "LATENT"), ("audio_latent", "LATENT")],
    "LTXVImgToVideo": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                       ("latent", "LATENT")],
    "LTXVAddGuide": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                     ("latent", "LATENT")],
    "LTXVCropGuides": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                       ("latent", "LATENT")],
    "LTXVLatentUpsampler": [("LATENT", "LATENT")],
    "LatentUpscaleModelLoader": [("LATENT_UPSCALE_MODEL",
                                  "LATENT_UPSCALE_MODEL")],
    "LoadImage": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    "LoadImageMask": [("MASK", "MASK")],
    "JoinImageWithAlpha": [("IMAGE", "IMAGE")],
    "LoadVideo": [("VIDEO", "VIDEO")],
    "LoadAudio": [("AUDIO", "AUDIO")],
    # All five slots, though no shipped graph wires the last two: slot
    # indices are positional, so a short list renumbers every wire after it.
    "GetVideoComponents": [("images", "IMAGE"), ("audio", "AUDIO"),
                           ("fps", "FLOAT"), ("bit_depth", "COMBO"),
                           ("color_space", "COMBO")],
    "GetImageSize": [("width", "INT"), ("height", "INT"),
                     ("batch_size", "INT")],
    "ImageFromBatch": [("IMAGE", "IMAGE")],
    "ImageScale": [("IMAGE", "IMAGE")],
    "ImageToMask": [("MASK", "MASK")],
    "MaskToImage": [("IMAGE", "IMAGE")],
    "LatentCutToBatch": [("LATENT", "LATENT")],
    "VAEEncode": [("LATENT", "LATENT")],
    "LTXVScheduler": [("SIGMAS", "SIGMAS")],
    "ManualSigmas": [("SIGMAS", "SIGMAS")],
    "Ideogram4Scheduler": [("SIGMAS", "SIGMAS")],
    "KSamplerSelect": [("SAMPLER", "SAMPLER")],
    "RandomNoise": [("NOISE", "NOISE")],
    "VAELoader": [("VAE", "VAE")],
    "VAEDecode": [("IMAGE", "IMAGE")],
    "VAEDecodeTiled": [("IMAGE", "IMAGE")],
    "VAEDecodeAudio": [("AUDIO", "AUDIO")],
    "LTXVAudioVAEDecode": [("Audio", "AUDIO")],
    # Save nodes grew an output socket upstream (comfy b0f9e326, 2026-06-22),
    # so RETURN_TYPES is ("IMAGE",) and RETURN_NAMES is ("images",). Nothing
    # wires it here, but the slot is positional, so declare it.
    "SaveImage": [("images", "IMAGE")],
    "SaveWEBM": [("images", "IMAGE")],
    "CreateVideo": [("VIDEO", "VIDEO")],
    "SaveVideo": [("video", "VIDEO")],
    "SaveAudioAdvanced": [("audio", "AUDIO")],
    "Note": [],
}


@_with_comfy_stubs
def _widget_options(node_type: str, wname: str) -> dict:
    if node_type in STOCK:
        return {}
    declared = NODE_CLASS_MAPPINGS[node_type].INPUT_TYPES()
    for section in ("required", "optional"):
        spec = declared.get(section, {}).get(wname)
        if isinstance(spec, tuple) and len(spec) > 1 and isinstance(spec[1], dict):
            return spec[1]
    return {}


def _takes_control_widget(node_type: str, wname: str) -> bool:
    """Does the frontend attach a control widget to this seed? Mirrors
    useIntWidget: a declared control_after_generate decides on its own
    truthiness, and an input that declares nothing falls back to its name."""
    options = _widget_options(node_type, wname)
    if CONTROL_WIDGET in options:
        return bool(options[CONTROL_WIDGET])
    return wname in SEED_WIDGET_NAMES


@_with_comfy_stubs
def _widget_names(node_type: str) -> list[str]:
    """Widget slots in frontend order, control widgets included."""
    if node_type in STOCK:
        declared = list(STOCK[node_type])
    else:
        cls = NODE_CLASS_MAPPINGS[node_type]
        declared = []
        for section in ("required", "optional"):
            for name, spec in cls.INPUT_TYPES().get(section, {}).items():
                kind = spec[0] if isinstance(spec, tuple) else spec
                if isinstance(kind, list) or kind in WIDGET_TYPES:
                    declared.append(name)
    names = []
    for name in declared:
        names.append(name)
        if _takes_control_widget(node_type, name):
            names.append(CONTROL_WIDGET)
    return names


@_with_comfy_stubs
def _connection_inputs(node_type: str) -> list[tuple[str, str]]:
    if node_type in STOCK:
        # Optional slots count: they hold their position in the slot index,
        # so leaving them out would renumber every wire after them.
        return {"CLIPTextEncode": [("clip", "CLIP")],
                "TextEncodeMageFlowEdit": [("clip", "CLIP"),
                                            ("images.image_1", "IMAGE"),
                                            ("images.image_2", "IMAGE"),
                                            ("images.image_3", "IMAGE"),
                                            ("images.image_4", "IMAGE"),
                                            ("vae", "VAE")],
                "TextEncodeQwenImage21": [
                    ("clip", "CLIP"), ("vae", "VAE"),
                    *[(f"images.image_{i}", "IMAGE") for i in range(1, 17)],
                ],
                "QwenImage21Cache": [("model", "MODEL")],
                "CLIPTextEncodeKandinsky5": [("clip", "CLIP")],
                "CLIPVisionEncode": [("clip_vision", "CLIP_VISION"),
                                     ("image", "IMAGE")],
                "ConditioningZeroOut": [("conditioning", "CONDITIONING")],
                "FluxGuidance": [("conditioning", "CONDITIONING")],
                "PiDConditioning": [("positive", "CONDITIONING"),
                                    ("latent", "LATENT")],
                "InstructPixToPixConditioning": [("positive", "CONDITIONING"),
                                                 ("negative", "CONDITIONING"),
                                                 ("vae", "VAE"),
                                                 ("pixels", "IMAGE")],
                "GetVideoComponents": [("video", "VIDEO")],
                "GetImageSize": [("image", "IMAGE")],
                "ImageFromBatch": [("image", "IMAGE")],
                "ImageScale": [("image", "IMAGE")],
                "ImageToMask": [("image", "IMAGE")],
                "MaskToImage": [("mask", "MASK")],
                "LatentCutToBatch": [("samples", "LATENT")],
                "VAEEncode": [("pixels", "IMAGE"), ("vae", "VAE")],
                "LTXVConditioning": [("positive", "CONDITIONING"),
                                     ("negative", "CONDITIONING")],
                "LTXVEmptyLatentAudio": [("audio_vae", "VAE")],
                "LTXVConcatAVLatent": [("video_latent", "LATENT"),
                                       ("audio_latent", "LATENT")],
                "LTXVSeparateAVLatent": [("av_latent", "LATENT")],
                "LTXVImgToVideo": [("positive", "CONDITIONING"),
                                   ("negative", "CONDITIONING"),
                                   ("vae", "VAE"), ("image", "IMAGE")],
                "JoinImageWithAlpha": [("image", "IMAGE"), ("alpha", "MASK")],
                "LTXVAddGuide": [("positive", "CONDITIONING"),
                                 ("negative", "CONDITIONING"),
                                 ("vae", "VAE"), ("latent", "LATENT"),
                                 ("image", "IMAGE"),
                                 ("attention_mask", "MASK"),
                                 ("iclora_parameters", "IC_LORA_PARAMETERS")],
                "LTXVCropGuides": [("positive", "CONDITIONING"),
                                   ("negative", "CONDITIONING"),
                                   ("latent", "LATENT")],
                "LTXVLatentUpsampler": [("samples", "LATENT"),
                                        ("upscale_model",
                                         "LATENT_UPSCALE_MODEL"),
                                        ("vae", "VAE")],
                "HunyuanRefinerLatent": [("positive", "CONDITIONING"),
                                         ("negative", "CONDITIONING"),
                                         ("latent", "LATENT")],
                "LTXVScheduler": [("latent", "LATENT")],
                "MiniMaxH3ImageToVideo": [("clip", "CLIP"), ("vae", "VAE"),
                                          ("first_frame", "IMAGE"), ("last_frame", "IMAGE")],
                # Another required-then-optional reorder: the schema declares
                # vae and audio_vae between positive and latent, but both are
                # optional=True, so create_input_dict_v1 files them under
                # `optional` and the frontend's transformNodeDefV1ToV2 walks
                # required first. latent is required, so it takes slot 1.
                "MiniMaxH3AddGuide": [("positive", "CONDITIONING"),
                                      ("latent", "LATENT"),
                                      ("vae", "VAE"), ("audio_vae", "VAE"),
                                      ("image", "IMAGE"), ("audio", "AUDIO")],
                "WanSCAILToVideo": [("positive", "CONDITIONING"),
                                    ("negative", "CONDITIONING"),
                                    ("vae", "VAE"), ("pose_video", "IMAGE"),
                                    ("pose_video_mask", "IMAGE"),
                                    ("reference_image", "IMAGE"),
                                    ("reference_image_mask", "IMAGE"),
                                    ("clip_vision_output", "CLIP_VISION_OUTPUT"),
                                    ("previous_frames", "IMAGE")],
                "WanImageToVideo": [("positive", "CONDITIONING"),
                                    ("negative", "CONDITIONING"),
                                    ("vae", "VAE"),
                                    ("clip_vision_output", "CLIP_VISION_OUTPUT"),
                                    ("start_image", "IMAGE")],
                "WanAnimate2ToVideo": [
                    ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                    ("vae", "VAE"), ("reference_image", "IMAGE"),
                    ("pose_video", "IMAGE"),
                    ("clip_vision_output", "CLIP_VISION_OUTPUT"),
                    ("positive_pose", "CONDITIONING"),
                    ("clip_vision_output_pose", "CLIP_VISION_OUTPUT"),
                    ("continue_motion", "IMAGE"),
                    # V3 moves linked scalar widgets after the optional media
                    # sockets. The pinned blueprint wires dimensions here.
                    ("width", "INT"), ("height", "INT"),
                    # A duplicate Animate2 chunk wires the prior node's
                    # output here. Keep this primitive socket visible even
                    # when its first-chunk value is still the widget default.
                    ("video_frame_offset", "INT")],
                # trim_amount is a widget until it receives the native
                # Animate2 trim-latent output; the legacy graph keeps the
                # connected primitive as a socket at this same position.
                "TrimVideoLatent": [("samples", "LATENT"), ("trim_amount", "INT")],
                "WanDancerEncodeAudio": [("audio", "AUDIO")],
                "WanDancerVideo": [("positive", "CONDITIONING"),
                                   ("negative", "CONDITIONING"),
                                   ("vae", "VAE"),
                                   ("clip_vision_output", "CLIP_VISION_OUTPUT"),
                                   ("clip_vision_output_ref", "CLIP_VISION_OUTPUT"),
                                   ("start_image", "IMAGE"), ("mask", "MASK"),
                                   ("audio_encoder_output", "AUDIO_ENCODER_OUTPUT")],
                "WanDancerPadKeyframesList": [("images", "IMAGE"),
                                              ("audio", "AUDIO")],
                # reference_images must stay absent. The schema declares it
                # between reference_video and ref_max_size, but as an autogrow
                # input (COMFY_AUTOGROW_V3) with min 0, so a freshly built node
                # carries no socket for it at all: the frontend draws an add
                # control and expands one IMAGE socket per press, serialized as
                # `reference_images.reference_image_N`. Listing a literal
                # `reference_images` socket here would emit an input line no real
                # graph carries. Wiring the in-context path means adding those
                # expanded names, which land after reference_video and renumber
                # nothing above them.
                "BerniniConditioning": [("positive", "CONDITIONING"),
                                        ("negative", "CONDITIONING"),
                                        ("vae", "VAE"),
                                        ("source_video", "IMAGE"),
                                        ("reference_video", "IMAGE")],
                "VAEDecode": [("samples", "LATENT"), ("vae", "VAE")],
                "VAEDecodeTiled": [("samples", "LATENT"), ("vae", "VAE")],
                "VAEDecodeAudio": [("samples", "LATENT"), ("vae", "VAE")],
                "LTXVAudioVAEDecode": [("samples", "LATENT"),
                                       ("audio_vae", "VAE")],
                "SaveImage": [("images", "IMAGE")],
                "SaveWEBM": [("images", "IMAGE"), ("fps", "FLOAT")],
                "CreateVideo": [("images", "IMAGE"), ("fps", "FLOAT"),
                                ("audio", "AUDIO")],
                "SaveVideo": [("video", "VIDEO")],
                "SaveAudioAdvanced": [("audio", "AUDIO")]}.get(node_type, [])
    cls = NODE_CLASS_MAPPINGS[node_type]
    declared = cls.INPUT_TYPES()
    out = []
    for section in ("required", "optional"):
        for name, spec in declared.get(section, {}).items():
            kind = spec[0] if isinstance(spec, tuple) else spec
            if not isinstance(kind, list) and kind not in WIDGET_TYPES:
                out.append((name, kind))
    return out


def _outputs(node_type: str) -> list[tuple[str, str]]:
    if node_type in STOCK_OUTPUTS:
        return STOCK_OUTPUTS[node_type]
    cls = NODE_CLASS_MAPPINGS[node_type]
    names = getattr(cls, "RETURN_NAMES", None) or cls.RETURN_TYPES
    return list(zip(names, cls.RETURN_TYPES, strict=False))


def _workflow_id(title: str) -> str:
    """Return the legacy 0.4 graph ID without WidgetId's reserved delimiter."""
    # ComfyUI Nodes 2 keys every widget as ``graphId:nodeId:name`` and inserts
    # the graph ID verbatim, so a colon kept from a title makes every widget key
    # unparseable. Drop only the colon: the other title-derived IDs stay as
    # they shipped.
    return title.lower().replace(" ", "-").replace(":", "")


@_with_comfy_stubs
def build(spec: dict) -> dict:
    """spec: {title, nodes: [{key, type, widgets:{name:value}, wire:{input: (key, out_idx)},
    pos:(col,row), note?}]} -> UI-format workflow dict."""
    nodes_out, links = [], []
    key_to_id = {n["key"]: i + 1 for i, n in enumerate(spec["nodes"])}
    link_id = 0
    node_links: dict = {}

    for n in spec["nodes"]:
        nid = key_to_id[n["key"]]
        for inp, (src_key, out_idx) in (n.get("wire") or {}).items():
            link_id += 1
            slot = next(i for i, (name, _) in enumerate(_connection_inputs(n["type"]))
                        if name == inp)
            links.append([link_id, key_to_id[src_key], out_idx, nid, slot,
                          dict(_connection_inputs(n["type"]))[inp]])
            node_links.setdefault((key_to_id[src_key], out_idx), []).append(link_id)
            node_links[("in", nid, inp)] = link_id

    for order, n in enumerate(spec["nodes"]):
        nid = key_to_id[n["key"]]
        ntype = n["type"]
        col, row = n["pos"]
        widgets = n.get("widgets", {})
        if ntype == "Note":
            wv = [n.get("note", "")]
        else:
            wv = []
            for wname in _widget_names(ntype):
                if wname == CONTROL_WIDGET:
                    # Ship every seed advancing. The seed value still holds,
                    # so the first render of a fresh template is the one the
                    # card shows.
                    wv.append(CONTROL_VALUE)
                elif wname in widgets:
                    wv.append(widgets[wname])
                else:
                    wv.append(_default_for(ntype, wname))
        inputs = [{"name": name, "type": kind,
                   "link": node_links.get(("in", nid, name))}
                  for name, kind in _connection_inputs(ntype)]
        outputs = [{"name": name, "type": kind, "slot_index": i,
                    "links": node_links.get((nid, i), [])}
                   for i, (name, kind) in enumerate(_outputs(ntype))]
        node_out = {
            "id": nid, "type": ntype,
            "pos": [80 + col * 340, 80 + row * 190],
            "size": [315, max(80, 30 + 22 * (len(inputs) + len(outputs) + len(wv)))],
            "flags": {}, "order": order, "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": ntype},
            "widgets_values": wv,
        }
        if ntype == "Note":
            title = spec["title"].removeprefix("DGX Monarch ")
            node_out["title"] = f"DGX Monarch | {title}"
        if ntype in NODE_CLASS_MAPPINGS:
            # This is the name-to-value map web/js/dgx_monarch_widgets.js saves,
            # so a widget added after release never sends a shipped template
            # through the positional heal meant for legacy saves.
            node_out["dgxm_widgets"] = dict(zip(_widget_names(ntype), wv, strict=True))
        nodes_out.append(node_out)
    return {"id": _workflow_id(spec["title"]), "revision": 0,
            "last_node_id": len(nodes_out), "last_link_id": link_id,
            "nodes": nodes_out, "links": links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4}


@_with_comfy_stubs
def _default_for(ntype: str, wname: str):
    if ntype in STOCK:
        return {"width": 1024, "height": 1024, "batch_size": 1, "text": "",
                "type": "flux", "device": "default", "length": 124, "format": "flac",
                "clip_l": "", "qwen25_7b": "", "frame_rate": 24.0, "steps": 20,
                "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1,
                "mu": 0.0, "std": 1.75, "noise_seed": 0, "sampler_name": "euler",
                "filename_prefix": "dgx_monarch"}.get(wname, "")
    cls = NODE_CLASS_MAPPINGS[ntype]
    declared = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        if wname in declared.get(section, {}):
            spec = declared[section][wname]
            kind = spec[0] if isinstance(spec, tuple) else spec
            opts = spec[1] if isinstance(spec, tuple) and len(spec) > 1 else {}
            if isinstance(kind, list):
                return opts.get("default", kind[0])
            return opts.get("default",
                            {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[kind])
    raise KeyError(f"{ntype}.{wname}")


FLUX_STACK = [
    {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
     "widgets": {"topology": "auto", "mode": "auto"}},
    {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
     "widgets": {"unet_name": "flux1-dev.safetensors"},
     "wire": {"mesh": ("init", 0)}},
    {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
     "widgets": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors",
                 "type": "flux"}},
    {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
     "widgets": {"text": "a lighthouse on a rocky coast at dusk, dramatic sky"},
     "wire": {"clip": ("clip", 0)}},
    {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 2),
     "wire": {"conditioning": ("pos", 0)}},
    {"key": "latent", "type": "EmptySD3LatentImage", "pos": (1, 3),
     "widgets": {"width": 1024, "height": 1024}},
    {"key": "vae", "type": "VAELoader", "pos": (3, 3), "widgets": {"vae_name": "ae.safetensors"}},
]


def _tail(sampler_key: str) -> list[dict]:
    return [
        {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
         "wire": {"samples": (sampler_key, 0), "vae": ("vae", 0)}},
        {"key": "save", "type": "SaveImage", "pos": (5, 1),
         "wire": {"images": ("dec", 0)}},
    ]


# Many template notes share these sentences. GATE states the first-render cost.
# DERIVED marks a graph read off the adapter and the stock node definitions,
# not replayed from a committed reference render. FRAMES explains the
# image-batch decode tail on the video graphs. Some notes repeat these words
# inline instead of using the constant; regenerate and diff after any edit.
GATE = ("The first render of a new model runs the identity gate, comparing the mesh "
        "with a stock single-GPU load before producing pixels. Allow a few minutes "
        "for proof renders. Later renders of the same combination skip the gate.")
NO_FIRST_USE_GATE = ("With slab, low-RSS and FSDP disabled, there is nothing for the "
                     "first-use gate to prove, so the first render skips it.")
DERIVED = ("This graph is derived from the family adapter and stock node definitions, "
           "without a committed reference render. Check filenames and shapes against "
           "your artifacts before running it.")
FRAMES = ("VAE Decode returns an image batch; Save Image writes numbered frames. "
          "For a video file, use the stock Create Video and Save Video nodes.")

# The Kandinsky 5 text encode node takes the prompt once per encoder, so each
# demo prompt is named once here: two written copies could drift apart.
_KANDINSKY5_IMAGE_PROMPT = (
    "A three-quarter side profile shot captured from a slightly low, stationary "
    "camera angle, this image frames a joyful hiker against the jagged, dramatic "
    "peaks of the Dolomites, where the elevated perspective emphasizes both the "
    "grandeur of the alpine landscape and the upward, hopeful tilt of his gaze. "
    "He wears a snug, mustard-yellow knit beanie that matches his chunky, "
    "textured sweater, paired with round, wire-rimmed glasses that add a "
    "thoughtful, approachable charm, while a rugged, oversized hiking backpack in "
    "weathered taupe is secured across his shoulders with gray, adjustable "
    "straps, complemented by a utility waist belt with a small, functional pouch. "
    "The scene is enhanced by a warm, vintage-inspired filter that bathes the "
    "frame in rich golden-amber tones, boosting contrast between the hiker's "
    "vibrant knitwear and the tawny mountain slopes, and a subtle film grain that "
    "lends a nostalgic, cinematic quality; soft, directional sunlight casts gentle "
    "shadows along his beard and sweater to add depth, with the crisp, saturated "
    "blue sky providing a striking counterpoint to the earthy foreground, creating "
    "an immersive portrait of adventure and warmth."
)
_KANDINSKY5_LITE_PROMPT = (
    "Rim light, side light, soft light, medium close-up, dusk, sunset, central "
    "composition, warm tones with low saturation, telephoto lens. A woman with "
    "fluffy brown curly hair stands elegantly in front of a magnificent stained "
    "glass window. She is wearing a flowing white dress, her hair neatly combed "
    "back, and her soft facial contours are gently illuminated by the colorful "
    "light filtering through the window from outside. The woman is talking to "
    "someone off-camera, yet there is a hint of sadness in her eyes, adding a "
    "layer of depth to her mysterious temperament. The background is dim, with a "
    "strong contrast between light and shadow, further emphasizing the tension of "
    "the character's emotions. The stained glass, under the glow of the setting "
    "sun, casts colorful light and shadows, enhancing the artistic sense and "
    "atmosphere of the overall picture."
)
_KANDINSKY5_PRO_PROMPT = (
    "Wide shot, slow dolly out, golden hour, low sun, side light, warm tones with "
    "low saturation, anamorphic lens. A wooden sailboat drifts across a calm bay "
    "while long shadows stretch over the water. The camera pulls back steadily to "
    "reveal a rocky headland crowned with pine trees, gulls crossing the frame, "
    "small waves catching the light. Fine spray lifts off the bow, the sail flexes "
    "in a light breeze, and the horizon stays level as the shot widens."
)


TEMPLATES = {
    "dgx-monarch-quickstart": {
        "title": "DGX Monarch quickstart",
        "nodes": [*FLUX_STACK,
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "DGX Monarch quickstart. It runs on one Spark as shipped (mode=auto). "
             "With a cluster.toml configured, topology=auto splits each render "
             "across the pair. The first render of a new model and LoRA "
             "combination runs the identity gate once (auto_gate)."},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
         *_tail("ks")],
    },
    "dgx-monarch-lora-low-rss": {
        "title": "LoRA stack with low-RSS mode",
        "nodes": [*FLUX_STACK,
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LoRA stack on unified memory. lora_low_rss=auto turns low-RSS mode on "
             "for DGX Spark: it frees comfy's backup of the original weights, about "
             "the size of the LoRA-covered model, and stack changes stay exact. A "
             "stack change restores the weights from the checkpoint bytes and bakes "
             "them again with the same values in seconds. Replace the placeholder LoRA "
             "names with your own."},
            {"key": "lora0", "type": "DGXMonarchLoraLoader", "pos": (2, 0),
             "widgets": {"lora_name": "your_first_lora.safetensors", "strength_model": 0.8},
             "wire": {"model": ("unet", 0)}},
            {"key": "lora1", "type": "DGXMonarchLoraLoader", "pos": (2, 1),
             "widgets": {"lora_name": "your_second_lora.safetensors", "strength_model": 0.5},
             "wire": {"model": ("lora0", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 1.0},
             "wire": {"model": ("lora1", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
         *_tail("ks")],
    },
    "dgx-monarch-fleet": {
        "title": "Fleet: many prompts, one per GPU",
        "nodes": [*FLUX_STACK,
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Fleet mode: each line in `prompts` becomes its own render on the next "
             "free GPU, with stock single-GPU math and no NCCL. The output is one "
             "latent batch in prompt order, and line i renders with seed "
             "noise_seed+i. For per-prompt weighted conditioning, wire a "
             "CONDITIONING list into `positive` instead."},
            {"key": "fleet", "type": "DGXMonarchFleetKSampler", "pos": (3, 1),
             "widgets": {"prompts": "a lighthouse on a rocky coast at dusk\n"
                                    "a wolf in a snowy forest, moonlight\n"
                                    "a desert caravan at golden hour\n"
                                    "a rainy neon street, reflections",
                         "steps": 20, "cfg": 1.0},
             "wire": {"model": ("unet", 0), "clip": ("clip", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
         *_tail("fleet")],
    },
    "dgx-monarch-identity-gate": {
        "title": "Identity Gate: prove your stack swaps exactly",
        "nodes": [*FLUX_STACK,
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "The Identity Gate renders your workflow twice, once through "
             "swap-produced weights and once through a fresh stock load, and reports "
             "PASS or FAIL with the bit-level latent difference. 2 steps are as "
             "conclusive as 20. You rarely need this node: the first render of "
             "a new combination runs the gate on its own. This node is the on-demand "
             "and CI form; strict=true raises on any verdict but PASS."},
            {"key": "lora0", "type": "DGXMonarchLoraLoader", "pos": (2, 0),
             "widgets": {"lora_name": "your_first_lora.safetensors", "strength_model": 0.8},
             "wire": {"model": ("unet", 0)}},
            {"key": "gate", "type": "DGXMonarchIdentityGate", "pos": (3, 1),
             "widgets": {"steps": 2},
             "wire": {"model": ("lora0", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
         *_tail("gate")],
    },
    "dgx-monarch-dual-spark-split": {
        "title": "Dual Spark: one render split across the pair",
        "nodes": [dict(n) for n in FLUX_STACK] + [
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Two Sparks, one render: with cluster.toml configured (two IPs), "
             "topology=auto picks the split from the auto table by resolution and "
             "CFG. This graph renders Flux 1 Dev at 1536x1536 and cfg 1.0, because "
             "the checkpoint is guidance distilled, so auto picks sequence-parallel "
             "(Ulysses). cfg-parallel needs a CFG above 1.0, and auto picks it only "
             "below 1.2 MP on a family whose table has a cfg2 row (Chroma, "
             "LongCat). Nothing else changes from the one-box workflow. dgxm top "
             "shows one burst of rail traffic per denoise step."},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
         *_tail("ks")],
    },
    # Per-family templates. Two of these (krea2, minimax-h3-t2va) take their
    # graph shape, not every widget value, from the committed API reference in
    # tests/fixtures/workflows/. The rest are read off the family adapter and
    # the stock node definitions; most say so in their own note through DERIVED
    # (h3-guide, ltx-t2v and ideogram4-t2i do not).
    "dgx-monarch-krea2-t2i": {
        "title": "Krea 2 text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Krea 2 text to image at 1536x1536 through the turbo checkpoint: 8 steps "
             "at cfg 1.0 with euler_cfg_pp. topology=auto runs it on one Spark and "
             "splits it across the pair once a cluster.toml is configured. At 1.2 MP "
             "and above, which this canvas is, auto also turns SAGE on for an fp8 "
             "checkpoint, so the kernel the Init widget shows is replaced at "
             "dispatch. The Krea2 table in docs/VALIDATION.md records fp8 uly2 with "
             "SAGE as HW (2026-08-21); the 2026-10-06 uly2 comparison with equal output values ran "
             "Torch Flash, not SAGE. sageattention must be installed on every worker, "
             "because a first configure that cannot import it raises instead of "
             "falling back. The first render of a new model runs the identity gate: "
             "before any pixels, a few minutes of proof renders check that the mesh "
             "matches a stock single-GPU load. Later renders of the same combination "
             "skip it. Expects "
             "krea2_turbo_fp8_scaled.safetensors in models/diffusion_models, "
             "qwen3vl_4b_bf16.safetensors in models/text_encoders, and "
             "qwen_image_vae.safetensors in models/vae, the VAE the official Krea 2 "
             "graphs name. The gate fixture tests/fixtures/workflows/krea2_uly2_api.json "
             "names wan_2.1_vae.safetensors because it records what one gate run "
             "loaded. VAE decode runs after the sequence-parallel gather, so no "
             "cross-rank or NRMS result depends on which VAE is used."},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "krea2_turbo_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "krea2"}},
            # The vendor prompting guide asks for long, detailed prompts and
            # ships this paragraph as one of its own examples.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A tiny, russet-brown harvest mouse clings to a "
                                 "slender diagonal branch amid vibrant green lobed "
                                 "leaves and small round buds. The mouse has soft "
                                 "textured fur, glossy black eyes, a pink nose, fine "
                                 "whiskers, and delicate pink paws firmly gripping "
                                 "the wood. In this macro photograph, an extremely "
                                 "shallow depth of field sharply focuses on the "
                                 "animal's face. The deep green background dissolves "
                                 "into a smooth, creamy bokeh, illuminated by soft, "
                                 "diffused natural lighting that highlights the "
                                 "intricate details of the fur and foliage."},
             "wire": {"clip": ("clip", 0)}},
            # Krea 2 turbo is CFG distilled, so cfg stays at 1.0. The empty encode
            # still reaches the render: euler_cfg_pp evaluates the negative at cfg
            # 1.0 and steps along that result, so zeroing it out or dropping it
            # changes the pixels (docs/TROUBLESHOOTING.md #21).
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": ""}, "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyLatentImage", "pos": (2, 3),
             "widgets": {"width": 1536, "height": 1536}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 8, "cfg": 1.0, "sampler_name": "euler_cfg_pp"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "qwen_image_vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_krea2"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-minimax-h3-t2va": {
        "title": "MiniMax H3 text to video with sound",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "MiniMax H3 text to video with sound: one packed video and audio sequence "
             "is denoised once, then decoded twice, the frames through the video VAE "
             "and the soundtrack through the audio VAE. 1344x768, 124 frames (about "
             "5 s at 24 fps), 20 steps at cfg 1.0, because the released checkpoints "
             "are CFG distilled. Comfy caps this DiT at batch 1, so cfg-parallel and "
             "data-parallel refuse; topology=auto asks for sequence-parallel on a "
             "configured pair and single on one box. Connect an image to first_frame "
             "(and last_frame) to make this a keyframe render. The first render of a "
             "new model runs the identity gate: before any pixels, a few minutes of "
             "proof renders check that the mesh matches a stock single-GPU load. "
             "Expects "
             "minimax_h3_fl2va_pruned_int8_convrot.safetensors in "
             "models/diffusion_models, qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors in "
             "models/text_encoders, and minimax_h3_video_vae_fp16.safetensors plus "
             "minimax_h3_audio_vae_fp32.safetensors in models/vae."},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                         "type": "minimax"}},
            {"key": "vvae", "type": "VAELoader", "pos": (0, 3),
             "widgets": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
            {"key": "avae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
            # The video VAE encodes any keyframe here, so this node needs it even
            # on the text-only path where both image slots stay empty.
            # The family convention is a structured prompt, not a caption: a look
            # line, a scene overview, a timed storyboard covering the whole clip,
            # a camera line in the vendor's own vocabulary, an audio line, and the
            # exclusions folded into the positive. Dialogue goes in the <d> speaker
            # tag, which holds only the language tag and the spoken words.
            {"key": "cond", "type": "MiniMaxH3ImageToVideo", "pos": (1, 2),
             "widgets": {"prompt":
                         "Realistic live-action cinematic look, craft documentary: "
                         "practical film photography, a glassblower's studio at "
                         "night, anamorphic lens, shallow depth of field, warm "
                         "furnace light against cold blue windows, fine film grain, "
                         "dust drifting in the air, restrained grading.\n\n"
                         "Scene overview: a glassblower turns a glowing gather of "
                         "molten glass on a blowpipe at the furnace mouth, shapes it "
                         "with wet folded paper, and lifts the finished bowl into "
                         "the light. One continuous craft moment, calm and "
                         "precise.\n\n"
                         "Storyboard (each shot its own scene, clean cuts):\n"
                         "[0s-1.5s] Shot 1: close on the furnace mouth, orange light "
                         "spilling out, the blowpipe entering frame and turning "
                         "steadily.\n"
                         "[1.5s-3s] Shot 2: medium side angle at the bench, forearms "
                         "rolling the pipe on the rails, the gather glowing white at "
                         "its core. The glassblower, a calm low voice (S1), says "
                         "without looking up: <d>[English] Keep it turning, or it "
                         "falls.</d>\n"
                         "[3s-4s] Shot 3: close on wet folded paper pressed to the "
                         "glass, steam bursting off the surface, the shape tightening "
                         "under the hand.\n"
                         "[4s-5s] Shot 4: the finished bowl lifted toward the window, "
                         "cold daylight passing through the amber wall of the glass, "
                         "holding.\n\n"
                         "Camera: shot 1 pushes in with small amplitude at slow "
                         "speed, shot 2 holds a static shot, shot 3 tracks with the "
                         "hands, shot 4 tilts up with small amplitude.\n\n"
                         "Audio: the low roar of the furnace under everything, the "
                         "steady scrape of the pipe on the bench rails, a sharp hiss "
                         "of steam at 3s, one soft chime as the bowl settles at 4.5s, "
                         "a sparse low string bed throughout.\n\n"
                         "No text, subtitles, logos or watermarks, no cartoon or CG "
                         "rendering, keep the live-action texture.",
                         "width": 1344, "height": 768, "length": 124},
             "wire": {"clip": ("clip", 0), "vae": ("vvae", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (2, 2),
             "widgets": {"text": ""}, "wire": {"clip": ("clip", 0)}},
            # Keep width/height/length equal to the conditioning node's: they
            # describe the same packed sequence twice.
            {"key": "latent", "type": "EmptyMiniMaxH3LatentAV", "pos": (2, 3),
             "widgets": {"width": 1344, "height": 768, "length": 124}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 1.0, "sampler_name": "res_multistep"},
             "wire": {"model": ("unet", 0), "positive": ("cond", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vdec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vvae", 0)}},
            {"key": "adec", "type": "VAEDecodeAudio", "pos": (4, 2),
             "wire": {"samples": ("ks", 0), "vae": ("avae", 0)}},
            {"key": "simg", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_minimax_h3"},
             "wire": {"images": ("vdec", 0)}},
            {"key": "saud", "type": "SaveAudioAdvanced", "pos": (5, 2),
             "widgets": {"filename_prefix": "audio/dgx_monarch_minimax_h3",
                         "format": "flac"},
             "wire": {"audio": ("adec", 0)}},
        ],
    },
    "dgx-monarch-minimax-h3-guide": {
        "title": "MiniMax H3 with a guide anchored mid video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "MiniMax H3 with an image anchored partway through the video. Add Guide "
             "pins a still or a clip at any frame of the target, not only the first "
             "and last frames the image-to-video node reaches. Chain Add Guide nodes "
             "to anchor several frames; each takes the conditioning from the one "
             "before it. frame_idx counts pixel frames from the start, and a "
             "negative value counts back from the end. A multi-frame image batch is "
             "anchored as a clip and cropped to the model's legal clip lengths (5, "
             "22, 39 and so on); a batch under 5 frames uses its first image only. To "
             "anchor a soundtrack starting at the same frame, connect audio_vae and "
             "an audio input instead. The guide adds condition rows to the sequence "
             "the workers split, so it costs memory: a clip charges a whole frame "
             "grid per latent frame, which the capacity preflight prices before the "
             "model loads. The guide also encodes through the video VAE first, and "
             "that encode is what runs out on a 121.6 GiB box: on 2026-08-14 it took "
             "13.6 GiB for one frame at this canvas and over 94 GiB for five, which "
             "never completed. Load Image names its file, so the preflight opens it "
             "and counts frames: one frame renders, an animated file or a video is "
             "refused, and a real clip needs a smaller canvas "
             "(docs/TROUBLESHOOTING.md #82 and #83). Point Load Image at your own "
             "still; the shipped filename is ComfyUI's stock sample. Any node "
             "between the loader and the guide hides the file, so nothing is proved "
             "and the clip charge stands. 1344x768, 124 frames at 24 fps, 20 steps "
             "at cfg 1.0, because the released checkpoints are CFG distilled. Comfy "
             "caps this DiT at batch 1, so cfg-parallel and data-parallel refuse. "
             "Expects "
             "minimax_h3_fl2va_pruned_int8_convrot.safetensors in "
             "models/diffusion_models, qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors in "
             "models/text_encoders, and minimax_h3_video_vae_fp16.safetensors plus "
             "minimax_h3_audio_vae_fp32.safetensors in models/vae."},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                         "type": "minimax"}},
            {"key": "vvae", "type": "VAELoader", "pos": (0, 3),
             "widgets": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
            {"key": "avae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 5),
             "widgets": {"image": "example.png", "upload": "image"}},
            # Same structured shape as the text to video template, written so the
            # middle beat sits at 2.5s to 3.5s: frame_idx 62 is 2.58 s into a 124
            # frame clip, so the anchored still has a described moment to pin.
            {"key": "cond", "type": "MiniMaxH3ImageToVideo", "pos": (1, 2),
             "widgets": {"prompt":
                         "Realistic live-action cinematic look, nature short: "
                         "practical film photography, a shallow mountain stream in "
                         "late autumn, anamorphic lens, shallow depth of field, low "
                         "golden sun through bare branches, fine film grain, cool "
                         "shadows against warm highlights.\n\n"
                         "Scene overview: a paper lantern set adrift on the stream "
                         "floats downstream, catches for a moment against a mossy "
                         "stone at the middle of the clip, then turns free and "
                         "carries on out of frame.\n\n"
                         "Storyboard (each shot its own scene, clean cuts):\n"
                         "[0s-2.5s] Shot 1: low over the water, the lantern drifting "
                         "toward camera, its candle steady, fallen leaves turning on "
                         "the surface beside it.\n"
                         "[2.5s-3.5s] Shot 2: the lantern meets a mossy stone and "
                         "holds there, the flame leaning, water folding around the "
                         "wet paper.\n"
                         "[3.5s-5s] Shot 3: the current lifts it free, it turns once "
                         "and slides away downstream into the low sun, going "
                         "small.\n\n"
                         "Camera: shot 1 tracks with the lantern at slow speed, shot "
                         "2 holds a static shot, shot 3 pulls out with small "
                         "amplitude.\n\n"
                         "Audio: running water throughout, close and shallow, a soft "
                         "knock as the lantern meets the stone at 2.6s, wind moving "
                         "through dry leaves, a single sustained low note under the "
                         "last two seconds.\n\n"
                         "No text, subtitles, logos or watermarks, no cartoon or CG "
                         "rendering, keep the live-action texture.",
                         "width": 1344, "height": 768, "length": 124},
             "wire": {"clip": ("clip", 0), "vae": ("vvae", 0)}},
            # Keep width, height and length equal to the conditioning node's: both
            # describe the same packed sequence. Add Guide reads them off this latent
            # to resolve frame_idx and to size the guide.
            {"key": "latent", "type": "EmptyMiniMaxH3LatentAV", "pos": (2, 3),
             "widgets": {"width": 1344, "height": 768, "length": 124}},
            {"key": "guide", "type": "MiniMaxH3AddGuide", "pos": (2, 4),
             "widgets": {"frame_idx": 62},
             "wire": {"positive": ("cond", 0), "vae": ("vvae", 0),
                      "latent": ("latent", 0), "image": ("image", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (2, 2),
             "widgets": {"text": ""}, "wire": {"clip": ("clip", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 1.0, "sampler_name": "res_multistep"},
             "wire": {"model": ("unet", 0), "positive": ("guide", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vdec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vvae", 0)}},
            {"key": "adec", "type": "VAEDecodeAudio", "pos": (4, 2),
             "wire": {"samples": ("ks", 0), "vae": ("avae", 0)}},
            {"key": "simg", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_minimax_h3_guide"},
             "wire": {"images": ("vdec", 0)}},
            {"key": "saud", "type": "SaveAudioAdvanced", "pos": (5, 2),
             "widgets": {"filename_prefix": "audio/dgx_monarch_minimax_h3_guide",
                         "format": "flac"},
             "wire": {"audio": ("adec", 0)}},
        ],
    },
    "dgx-monarch-wan-t2v": {
        "title": "Wan text to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan 2.1 text to video at 832x480, 81 frames (about 5 s at 16 fps), "
             "20 steps at cfg 5.0 with uni_pc, the solver both official Wan 2.1 "
             "sources use. auto splits the token sequence, not the CFG batch, on "
             "this family (docs/MODELS.md) and runs the same graph on one Spark "
             "when no cluster.toml is configured. The SD3 sampling shift is render "
             "math on the resident model, so it goes through Model Sampling SD3 "
             "(DGX Monarch); the stock patch node cannot take a mesh model. Wan "
             "2.2's 14B mixture of experts is two checkpoints on one denoise "
             "schedule: open dgx-monarch-wan22-t2v or dgx-monarch-wan22-i2v "
             "instead of rebuilding this graph by hand, because those also run a "
             "different sampling shift and a second sampler stage. " + FRAMES + " " +
             GATE + " Expects wan2.1_t2v_14B_fp8_scaled.safetensors in "
             "models/diffusion_models, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, and wan_2.1_vae.safetensors in models/vae. " +
             DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "wan2.1_t2v_14B_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 8.0}, "wire": {"model": ("unet", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "a paper lantern drifting over a quiet harbour at "
                                 "night, reflections on the water, slow drift"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low quality, static, watermark, text, "
                                 "distorted"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyHunyuanLatentVideo", "pos": (2, 3),
             "widgets": {"width": 832, "height": 480, "length": 81, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             # sampler_name is explicit: the generator's comfy stub lists only
             # euler, so an omitted value ships euler with no error.
             "widgets": {"steps": 20, "cfg": 5.0, "sampler_name": "uni_pc"},
             "wire": {"model": ("shift", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan_t2v"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-wan21-i2v": {
        "title": "Wan 2.1 image to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan 2.1 image to video at 832x480, 81 frames (about 5 s at 16 fps), "
             "20 steps at cfg 6.0 with uni_pc. Point Load Image at your own first "
             "frame; the shipped filename is a placeholder. Wan Image To Video "
             "resizes the still to the widget size through a center crop, so keep "
             "the aspect ratio close, and write the prompt about what moves, "
             "because the subject comes from the image. That node encodes the "
             "frame into concat conditioning on both the positive and the "
             "negative, so neither may bypass it, and it builds the latent itself, "
             "so the graph has no Empty latent node. CLIP Vision Encode at crop "
             "none carries the same frame into the replicated cross-attention "
             "context, which Wan 2.1 image to video uses and Wan 2.2 image to video "
             "drops. auto splits the token sequence, not the CFG batch, on this "
             "family (docs/MODELS.md), so cfg above 1.0 changes quality and never "
             "adds a rank, and the same graph runs on one Spark when no "
             "cluster.toml is configured. The SD3 sampling shift is render math on "
             "the resident model, so it goes through Model Sampling SD3 (DGX "
             "Monarch); the stock patch node cannot take a mesh model. For the "
             "720p tier load wan2.1_i2v_720p_14B_fp8_scaled.safetensors and raise "
             "the canvas to match, and for the uly2+fsdp capacity preset load the "
             "file stored in BF16, wan2.1_i2v_480p_14B_bf16.safetensors. The scoped HW "
             "evidence covers the FP8-scaled auto/Ulysses 81-frame path: cold and "
             "warm renders completed, and its decoded one-step probe matched the "
             "local reference exactly. It does not cover this BF16 capacity "
             "variant, other quantizations, FSDP or unlisted workflows. " + FRAMES + " " +
             GATE + " Expects wan2.1_i2v_480p_14B_fp8_scaled.safetensors in "
             "models/diffusion_models, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, wan_2.1_vae.safetensors in models/vae, and "
             "clip_vision_h.safetensors in models/clip_vision. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "wan2.1_i2v_480p_14B_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 8.0}, "wire": {"model": ("unet", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "a market stall at first light, steam curling from "
                                 "a copper pot, the striped awning lifting in a slow "
                                 "breeze, crates of fruit catching warm low sun, a "
                                 "slow cinematic push-in, gentle natural motion, "
                                 "shallow depth of field"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low quality, static, watermark, text, "
                                 "distorted"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            {"key": "cvl", "type": "CLIPVisionLoader", "pos": (0, 5),
             "widgets": {"clip_name": "clip_vision_h.safetensors"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 6),
             "widgets": {"image": "example.png", "upload": "image"}},
            # crop none matches every official Wan 2.1 i2v source. The default,
            # center, would crop the frame square for CLIP Vision and cut edges
            # the 832x480 canvas keeps.
            {"key": "cve", "type": "CLIPVisionEncode", "pos": (1, 5),
             "widgets": {"crop": "none"},
             "wire": {"clip_vision": ("cvl", 0), "image": ("image", 0)}},
            # Width, height and length live here instead of on an Empty latent
            # node: this node builds the latent. It also writes concat_latent_image
            # and concat_mask onto both conditionings, so the negative has to come
            # through it too; a negative wired straight to the sampler reaches the
            # model with zeros in those channels and changes the render.
            {"key": "i2v", "type": "WanImageToVideo", "pos": (2, 3),
             "widgets": {"width": 832, "height": 480, "length": 81, "batch_size": 1},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0), "clip_vision_output": ("cve", 0),
                      "start_image": ("image", 0)}},
            # sampler_name is explicit for the reason the Wan text to video graph gives.
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 6.0, "sampler_name": "uni_pc"},
             "wire": {"model": ("shift", 0), "positive": ("i2v", 0),
                      "negative": ("i2v", 1), "latent_image": ("i2v", 2)}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan21_i2v"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-wan-animate2": {
        "title": "Wan-Animate 2 motion transfer",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use",
                         "slab_weights": "off", "lora_low_rss": "off"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan-Animate 2 Base motion transfer: one 81-frame chunk. The visible "
             "ImageScale (Upscale Image) node alone sets the output size; it defaults "
             "to portrait 480x832 for portrait reference and driving media. For a "
             "landscape character and driving clip, set it to 832x480 and use a "
             "landscape reference. Load Video is the driving motion and Load Image "
             "is the reference character; the reference prompt describes appearance "
             "and the pose prompt describes the driving action. Animate2 "
             "conditioning requires both CLIP Vision encodes and both prompt paths. "
             "The native conditioner resizes the raw driving frames to this canvas, "
             "and the export uses that loaded video's FPS, so playback speed does "
             "not change. To extend a clip, copy the Animate2 conditioning and "
             "sampling tail, then wire this chunk's decoded frames to its "
             "continue_motion input and this node's video_frame_offset output to its "
             "video_frame_offset input; each Queue press renders one 81-frame chunk. "
             "This is the current ComfyUI Base blueprint recipe: the supplied "
             "LightX2V LoRA at 1.0, LCM, six steps, and the serialized DGX SD3 shift "
             "5.0. The sampler needs cfg=1.0, which here applies no CFG guidance. The "
             "template defaults to the official INT8 ConvRot DiT to lower model "
             "memory; the Base-quality and Distilled templates keep their BF16 "
             "defaults. The distributed model stays resident after decode, so later "
             "renders reuse it. This recipe uses ordinary weight and LoRA residency, "
             "with slab and low-RSS off; the warm tests do not cover them. " + NO_FIRST_USE_GATE + " Stock "
             "WanAnimate2Cache and stock model-patch nodes take MODEL, not "
             "DGXM_MODEL, so cache stays off here. See docs/MODELS.md for the "
             "measured Base and Distilled BF16 recipes and the accelerated "
             "BF16/INT8+LightX2V six-step recipes. The separate Wan-Animate 2 "
             "memory-saver template recycles actors before decode, for when lower "
             "memory use is worth reloading the model later. Unlisted settings, "
             "broader continuation, cache, first-use LoRA, slab/low-RSS, FSDP, ring, "
             "and background replacement have no passing reference comparison. Expects "
             "wan_animate_2_int8_convrot.safetensors in models/diffusion_models, "
             "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors in "
             "models/loras, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, clip_vision_h.safetensors in models/clip_vision, "
             "and Wan2_1_VAE_bf16.safetensors in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "wan_animate_2_int8_convrot.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "lora", "type": "DGXMonarchLoraLoader", "pos": (2, 0),
             "widgets": {"lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
                         "strength_model": 1.0}, "wire": {"model": ("unet", 0)}},
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (3, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("lora", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "a dancer in an indigo jacket, cinematic natural light, detailed fabric"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blur, low quality, distorted anatomy, watermark, text"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "pose", "type": "CLIPTextEncode", "pos": (1, 4),
             "widgets": {"text": "a dancer makes a smooth turn, then steps forward with expressive arms"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
            {"key": "vision", "type": "CLIPVisionLoader", "pos": (0, 6),
             "widgets": {"clip_name": "clip_vision_h.safetensors"}},
            {"key": "reference", "type": "LoadImage", "pos": (0, 7),
             "widgets": {"image": "wan_animate2_reference.png", "upload": "image"}},
            {"key": "driving", "type": "LoadVideo", "pos": (0, 8),
             "widgets": {"file": "wan_animate2_driving.mp4", "upload": "image"}},
            {"key": "parts", "type": "GetVideoComponents", "pos": (1, 8),
             "wire": {"video": ("driving", 0)}},
            {"key": "driving_scale", "type": "ImageScale", "pos": (1, 9),
             "widgets": {"upscale_method": "lanczos", "width": 480, "height": 832,
                         "crop": "disabled"}, "wire": {"image": ("parts", 0)}},
            {"key": "canvas", "type": "GetImageSize", "pos": (2, 9),
             "wire": {"image": ("driving_scale", 0)}},
            {"key": "ref_vision", "type": "CLIPVisionEncode", "pos": (1, 6),
             "widgets": {"crop": "none"},
             "wire": {"clip_vision": ("vision", 0), "image": ("reference", 0)}},
            {"key": "pose_first", "type": "ImageFromBatch", "pos": (2, 8),
             "widgets": {"batch_index": 0, "length": 1}, "wire": {"image": ("driving_scale", 0)}},
            {"key": "pose_vision", "type": "CLIPVisionEncode", "pos": (2, 6),
             "widgets": {"crop": "none"},
             "wire": {"clip_vision": ("vision", 0), "image": ("pose_first", 0)}},
            {"key": "animate", "type": "WanAnimate2ToVideo", "pos": (3, 3),
             "widgets": {"width": 480, "height": 832, "length": 81, "batch_size": 1,
                         "video_frame_offset": 0, "pose_strength": 1.0,
                         "pose_start_percent": 0.0, "pose_end_percent": 1.0,
                         "reference_image_strength": 1.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0), "vae": ("vae", 0),
                      "width": ("canvas", 0), "height": ("canvas", 1),
                      "reference_image": ("reference", 0), "pose_video": ("driving_scale", 0),
                      "clip_vision_output": ("ref_vision", 0), "positive_pose": ("pose", 0),
                      "clip_vision_output_pose": ("pose_vision", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (4, 2),
             "widgets": {"steps": 6, "cfg": 1.0, "sampler_name": "lcm", "scheduler": "simple"},
             "wire": {"model": ("shift", 0), "positive": ("animate", 0),
                      "negative": ("animate", 1), "latent_image": ("animate", 2)}},
            {"key": "trim", "type": "TrimVideoLatent", "pos": (5, 2),
             "wire": {"samples": ("ks", 0), "trim_amount": ("animate", 3)}},
            {"key": "dec", "type": "VAEDecode", "pos": (6, 2),
             "wire": {"samples": ("trim", 0), "vae": ("vae", 0)}},
            {"key": "video", "type": "CreateVideo", "pos": (7, 2),
             "widgets": {"bit_depth": 8, "color_space": "sRGB", "codec": "none"},
             "wire": {"images": ("dec", 0), "audio": ("parts", 1), "fps": ("parts", 2)}},
            {"key": "save", "type": "SaveVideo", "pos": (8, 2),
             "widgets": {"filename_prefix": "video/dgx_monarch_wan_animate2", "format": "auto", "codec": "auto"},
             "wire": {"video": ("video", 0)}},
        ],
    },
    "dgx-monarch-wan-animate2-distilled": {
        "title": "Wan-Animate 2 Distilled motion transfer",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use",
                         "slab_weights": "off", "lora_low_rss": "off"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan-Animate 2 Distilled motion transfer: one 81-frame chunk. The visible "
             "ImageScale (Upscale Image) node defaults to portrait 480x832 for "
             "portrait reference and driving media; for a landscape composition, use "
             "832x480 with landscape media. Use your own raw driving video and "
             "reference image. The generation and pose prompts, the reference CLIP "
             "Vision encode and the CLIP Vision encode of the driving clip's first "
             "frame all stay in the native conditioning. The native node resizes "
             "frames spatially, and the export takes the loaded driving video's FPS. "
             "For another 81-frame chunk, copy the Animate2 tail, then wire the "
             "decoded frames to continue_motion and the prior video_frame_offset "
             "output to the next node. This is the pinned ComfyUI Distilled "
             "blueprint recipe: ten LCM steps, no LightX2V LoRA, and the serialized "
             "DGX SD3 shift 5.0. cfg=1.0 fills the sampler input and adds no CFG "
             "guidance. Cache stays off because native MODEL cache and patch nodes "
             "cannot take DGXM_MODEL. See docs/MODELS.md for the measured Base and "
             "Distilled BF16 recipes and the accelerated INT8+LightX2V six-step "
             "recipe. Unlisted settings, broader continuation, cache, first-use "
             "LoRA, slab/low-RSS, FSDP, ring, and background replacement remain "
             "without a passing reference comparison. This template ships slab and low-RSS off. " + NO_FIRST_USE_GATE +
             " Expects wan_animate_2_distill_bf16.safetensors in models/diffusion_models, "
             "umt5_xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders, "
             "clip_vision_h.safetensors in models/clip_vision, and "
             "Wan2_1_VAE_bf16.safetensors in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "wan_animate_2_distill_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("unet", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "a dancer in an indigo jacket, cinematic natural light, detailed fabric"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blur, low quality, distorted anatomy, watermark, text"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "pose", "type": "CLIPTextEncode", "pos": (1, 4),
             "widgets": {"text": "a dancer makes a smooth turn, then steps forward with expressive arms"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
            {"key": "vision", "type": "CLIPVisionLoader", "pos": (0, 6),
             "widgets": {"clip_name": "clip_vision_h.safetensors"}},
            {"key": "reference", "type": "LoadImage", "pos": (0, 7),
             "widgets": {"image": "wan_animate2_reference.png", "upload": "image"}},
            {"key": "driving", "type": "LoadVideo", "pos": (0, 8),
             "widgets": {"file": "wan_animate2_driving.mp4", "upload": "image"}},
            {"key": "parts", "type": "GetVideoComponents", "pos": (1, 8),
             "wire": {"video": ("driving", 0)}},
            {"key": "driving_scale", "type": "ImageScale", "pos": (1, 9),
             "widgets": {"upscale_method": "lanczos", "width": 480, "height": 832,
                         "crop": "disabled"}, "wire": {"image": ("parts", 0)}},
            {"key": "canvas", "type": "GetImageSize", "pos": (2, 9),
             "wire": {"image": ("driving_scale", 0)}},
            {"key": "ref_vision", "type": "CLIPVisionEncode", "pos": (1, 6),
             "widgets": {"crop": "none"},
             "wire": {"clip_vision": ("vision", 0), "image": ("reference", 0)}},
            {"key": "pose_first", "type": "ImageFromBatch", "pos": (2, 8),
             "widgets": {"batch_index": 0, "length": 1}, "wire": {"image": ("driving_scale", 0)}},
            {"key": "pose_vision", "type": "CLIPVisionEncode", "pos": (2, 6),
             "widgets": {"crop": "none"},
             "wire": {"clip_vision": ("vision", 0), "image": ("pose_first", 0)}},
            {"key": "animate", "type": "WanAnimate2ToVideo", "pos": (3, 3),
             "widgets": {"width": 480, "height": 832, "length": 81, "batch_size": 1,
                         "video_frame_offset": 0, "pose_strength": 1.0,
                         "pose_start_percent": 0.0, "pose_end_percent": 1.0,
                         "reference_image_strength": 1.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0), "vae": ("vae", 0),
                      "width": ("canvas", 0), "height": ("canvas", 1),
                      "reference_image": ("reference", 0), "pose_video": ("driving_scale", 0),
                      "clip_vision_output": ("ref_vision", 0), "positive_pose": ("pose", 0),
                      "clip_vision_output_pose": ("pose_vision", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (4, 2),
             "widgets": {"steps": 10, "cfg": 1.0, "sampler_name": "lcm", "scheduler": "simple"},
             "wire": {"model": ("shift", 0), "positive": ("animate", 0),
                      "negative": ("animate", 1), "latent_image": ("animate", 2)}},
            {"key": "trim", "type": "TrimVideoLatent", "pos": (5, 2),
             "wire": {"samples": ("ks", 0), "trim_amount": ("animate", 3)}},
            {"key": "dec", "type": "VAEDecode", "pos": (6, 2),
             "wire": {"samples": ("trim", 0), "vae": ("vae", 0)}},
            {"key": "video", "type": "CreateVideo", "pos": (7, 2),
             "widgets": {"bit_depth": 8, "color_space": "sRGB", "codec": "none"},
             "wire": {"images": ("dec", 0), "audio": ("parts", 1), "fps": ("parts", 2)}},
            {"key": "save", "type": "SaveVideo", "pos": (8, 2),
             "widgets": {"filename_prefix": "video/dgx_monarch_wan_animate2_distilled", "format": "auto", "codec": "auto"},
             "wire": {"video": ("video", 0)}},
        ],
    },
    "dgx-monarch-wan22-t2v": {
        "title": "Wan 2.2 text to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan 2.2 text to video at 832x480, 81 frames (about 5 s at 16 fps), on "
             "the 14B mixture of experts: 20 steps at cfg 3.5 split at step 10, the "
             "high-noise checkpoint first and the low-noise one second, with sampling "
             "shift 5.0 on both stages. Each stage is its own render against a store "
             "that holds one base per slot, so a queue run loads the high-noise "
             "checkpoint, drops it, and loads the low-noise one; the second stage "
             "continues from the leftover noise the first returns. auto splits the "
             "token sequence, not the CFG batch, on this family (docs/MODELS.md), "
             "and this canvas is 32760 tokens per stream, which halves with no pad. "
             "Both loaders set weight_dtype to bf16, because the row behind this "
             "template covers the BF16 high/low pair and leaves FP8 unclaimed; the "
             "official example ships "
             "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors and "
             "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors at the default dtype "
             "instead, which is half the download and leaves every number above "
             "unchanged. For the four-step lightx2v path, add a Load LoRA (DGX "
             "Monarch) node after each UNET loader with "
             "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors and "
             "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors at strength "
             "1.0, then set 4 steps, cfg 1.0, and the split at step 2. " + FRAMES +
             " " + GATE + " Expects wan2.2_t2v_high_noise_14B_fp16.safetensors plus "
             "wan2.2_t2v_low_noise_14B_fp16.safetensors in models/diffusion_models, "
             "umt5_xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders, and "
             "wan_2.1_vae.safetensors in models/vae. " + DERIVED},
            # Two loaders, one Init: the MoE's high-noise and low-noise experts are
            # two checkpoints on one denoise schedule, not a base and a refiner.
            {"key": "high", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "wan2.2_t2v_high_noise_14B_fp16.safetensors",
                         "weight_dtype": "bf16"},
             "wire": {"mesh": ("init", 0)}},
            # The SD3 sampling shift patches the resident model, so it goes through
            # the DGX Monarch node; each stage patches its own model at 5.0.
            {"key": "hshift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("high", 0)}},
            {"key": "low", "type": "DGXMonarchUNETLoader", "pos": (1, 1),
             "widgets": {"unet_name": "wan2.2_t2v_low_noise_14B_fp16.safetensors",
                         "weight_dtype": "bf16"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "lshift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 1),
             "widgets": {"shift": 5.0}, "wire": {"model": ("low", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "Two anthropomorphic cats in comfy boxing gear and "
                                 "bright gloves fight intensely on a spotlighted "
                                 "stage."},
             "wire": {"clip": ("clip", 0)}},
            # Wan ships its negative as one Chinese boilerplate string and both
            # official 14B templates carry it; this is that string verbatim. The
            # text to video variant keeps the tail the image to video one drops.
            # RUF001 flags the full-width commas as lookalikes of ASCII; they are
            # the real characters.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text":
                         "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"  # noqa: RUF001
                         "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"  # noqa: RUF001
                         "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"  # noqa: RUF001
                         "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"  # noqa: RUF001
                         "三条腿，背景人很多，倒着走，裸露，NSFW"},  # noqa: RUF001
             "wire": {"clip": ("clip", 0)}},
            # 832x480 with 81 frames is 21 latent frames over a /16 token grid:
            # 21 * 30 * 52 = 32760 rows, even, so neither rank attends a pad row.
            {"key": "latent", "type": "EmptyHunyuanLatentVideo", "pos": (2, 3),
             "widgets": {"width": 832, "height": 480, "length": 81, "batch_size": 1}},
            # High stage: adds the noise, stops at the boundary and hands the
            # leftover noise on. Its end_at_step and the low stage's start_at_step
            # are the same number, and both stages carry the full step count.
            {"key": "ks_high", "type": "DGXMonarchKSamplerAdvanced", "pos": (3, 0),
             "widgets": {"add_noise": "enable", "steps": 20, "cfg": 3.5,
                         "sampler_name": "euler", "scheduler": "simple",
                         "start_at_step": 0, "end_at_step": 10,
                         "return_with_leftover_noise": "enable"},
             "wire": {"model": ("hshift", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            # Low stage: takes the high stage's latent, adds no noise of its own and
            # finishes the schedule. end_at_step is written as the step count rather
            # than left at the node's 10000 default, matching the official graph.
            {"key": "ks_low", "type": "DGXMonarchKSamplerAdvanced", "pos": (3, 1),
             "widgets": {"add_noise": "disable", "steps": 20, "cfg": 3.5,
                         "sampler_name": "euler", "scheduler": "simple",
                         "start_at_step": 10, "end_at_step": 20,
                         "return_with_leftover_noise": "disable"},
             "wire": {"model": ("lshift", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("ks_high", 0)}},
            # The 14B checkpoints use the Wan 2.1 VAE; wan2.2_vae.safetensors
            # belongs to the 5B TI2V model.
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks_low", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan22_t2v"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-wan22-i2v": {
        "title": "Wan 2.2 image to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan 2.2 image to video at 640x640, 81 frames (about 5 s at 16 fps). "
             "The 14B mixture of experts is two checkpoints on one schedule: the "
             "high-noise stage runs steps 0 to 10 and hands its leftover noise to "
             "the low-noise stage, which finishes 10 to 20, both at cfg 3.5 with "
             "euler, simple, and SD3 sampling shift 5.0. Each stage is its own "
             "render, and the one-base-per-slot store drops one checkpoint to load "
             "the other, so one queue run loads twice (docs/MODELS.md). auto splits "
             "the token sequence, not the CFG batch, on this family and runs the "
             "same graph on one Spark when no cluster.toml is configured; 640x640 "
             "with 21 latent frames gives 33600 tokens per stream, which halves at "
             "world 2 with no pad. Point Load Image at your own first frame, because "
             "the shipped filename is a placeholder, and leave clip_vision_output "
             "empty: Wan 2.2 image to video needs no CLIP Vision encode. For the "
             "four-step lightx2v path, add a LoRA loader after each UNET loader with "
             "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors and "
             "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors from "
             "models/loras at strength 1.0, then set steps 4, the boundary at 2, and "
             "cfg 1.0, and leave sampler, scheduler and shift alone. The hardware "
             "row records that four-step path at 640x640 with 17 decoded frames; "
             "this 20-step default is outside it. " + FRAMES + " " +
             GATE + " Expects wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors plus "
             "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors in "
             "models/diffusion_models, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, and wan_2.1_vae.safetensors in models/vae. " +
             DERIVED},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "example.png", "upload": "image"}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            # Two checkpoints, one Init: the store swaps the resident base between
            # the two sampler stages.
            {"key": "unet_high", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "unet_low", "type": "DGXMonarchUNETLoader", "pos": (1, 1),
             "widgets": {"unet_name":
                         "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "the subject holds still while the scene comes alive "
                                 "around it, the camera pushes in slowly and drifts "
                                 "left, soft natural light, gentle atmospheric "
                                 "motion, cinematic depth of field"},
             "wire": {"clip": ("clip", 0)}},
            # This is the Wan negative from the Wan 2.2 text to video graph, cut where
            # the official image to video template cuts it: at the walking backwards
            # clause, with no NSFW tail.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text":
                         "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"  # noqa: RUF001
                         "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"  # noqa: RUF001
                         "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"  # noqa: RUF001
                         "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"  # noqa: RUF001
                         "三条腿，背景人很多，倒着走"},  # noqa: RUF001
             "wire": {"clip": ("clip", 0)}},
            # Each expert gets its own shift node at the official 5.0, not the
            # 8.0 the Wan 2.1 template uses.
            {"key": "shift_high", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("unet_high", 0)}},
            {"key": "shift_low", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 1),
             "widgets": {"shift": 5.0}, "wire": {"model": ("unet_low", 0)}},
            # Wan Image To Video builds the latent itself and writes the first frame
            # into both conditionings, so both stages take positive and negative from
            # here. clip_vision_output stays empty: Wan 2.2 i2v needs no CLIP Vision.
            {"key": "cond", "type": "WanImageToVideo", "pos": (2, 3),
             "widgets": {"width": 640, "height": 640, "length": 81, "batch_size": 1},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0), "start_image": ("image", 0)}},
            # The two stages split the schedule as in the Wan 2.2 text to video graph.
            {"key": "ks_high", "type": "DGXMonarchKSamplerAdvanced", "pos": (3, 0),
             "widgets": {"add_noise": "enable", "steps": 20, "cfg": 3.5,
                         "sampler_name": "euler", "scheduler": "simple",
                         "start_at_step": 0, "end_at_step": 10,
                         "return_with_leftover_noise": "enable"},
             "wire": {"model": ("shift_high", 0), "positive": ("cond", 0),
                      "negative": ("cond", 1), "latent_image": ("cond", 2)}},
            {"key": "ks_low", "type": "DGXMonarchKSamplerAdvanced", "pos": (3, 1),
             "widgets": {"add_noise": "disable", "steps": 20, "cfg": 3.5,
                         "sampler_name": "euler", "scheduler": "simple",
                         "start_at_step": 10, "end_at_step": 20,
                         "return_with_leftover_noise": "disable"},
             "wire": {"model": ("shift_low", 0), "positive": ("cond", 0),
                      "negative": ("cond", 1), "latent_image": ("ks_high", 0)}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks_low", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan22_i2v"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-wan-flowrvs": {
        "title": "Wan FlowRVS referring video segmentation",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan FlowRVS is referring video object segmentation, not video "
             "generation: it takes your clip and a short phrase naming one object "
             "and returns a mask of that object for each frame, here at 832x480 "
             "over 17 frames, one euler step on the simple schedule at cfg 1.0, "
             "starting from the VAE encode of the source frames with no added "
             "noise. ComfyUI ships no example clip, so the shipped Load Video "
             "filename is a placeholder: upload your own clip, point the widget at "
             "it, give it at least 17 frames, and keep any frame count you change on "
             "the 4k+1 grid the Wan VAE needs (17, 33, 81; the official example "
             "uses 81). The prompt is a referring expression, so keep it short and "
             "name one object. The negative is the positive zeroed out, because cfg "
             "1.0 evaluates no second branch, and the seed has no effect, because "
             "the sampler discards its noise before the first forward. One VAE "
             "loader serves both ends: the mask VAE keeps the stock Wan encoder and "
             "changes only the decoder, which writes a single channel. So the tail "
             "passes through Convert Image to Mask and Convert Mask to Image before "
             "Save Image, the stock tiled decode cannot read this file, and the "
             "official way to present a binary mask or an overlay is stock Threshold "
             "Mask at 0.5 with Image Composite Masked over the source frames. "
             "Detection reads the checkpoint's safetensors metadata, not its tensor "
             "shapes, so a repack or requantize that drops the metadata loads as a "
             "plain Wan 2.1 text to video model and renders nothing useful. "
             + FRAMES + " " + GATE + " Expects "
             "wan21_1.3b_flow_rvs_bf16.safetensors in models/diffusion_models, "
             "umt5_xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders, and "
             "wan21_flow_rvs_mask_vae_bf16.safetensors in models/vae. Every tensor "
             "in that checkpoint is bf16, so set the loader's weight_dtype to bf16 "
             "to keep that precision instead of the default's choice. Tests cover "
             "BF16 synthetic-clip auto/Ulysses: cold and warm renders "
             "completed, and the decoded 17-frame one-step probe matched its local "
             "reference exactly. Real-media segmentation quality, other recipes and "
             "FSDP have no passing reference comparison. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "wan21_1.3b_flow_rvs_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "the person in the red jacket walking toward the "
                                 "camera"},
             "wire": {"clip": ("clip", 0)}},
            # cfg 1.0 evaluates one branch, so a typed negative would never be
            # read. The official graph zeroes the positive instead.
            {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 2),
             "wire": {"conditioning": ("pos", 0)}},
            {"key": "video", "type": "LoadVideo", "pos": (0, 3),
             "widgets": {"file": "source_video.mp4", "upload": "image"}},
            {"key": "comp", "type": "GetVideoComponents", "pos": (1, 3),
             "wire": {"video": ("video", 0)}},
            # 17 frames on the 4k+1 grid, then 832x480 so both axes divide by 16
            # and the 7800-token grid shards by 2 with no pad.
            {"key": "frames", "type": "ImageFromBatch", "pos": (2, 3),
             "widgets": {"batch_index": 0, "length": 17},
             "wire": {"image": ("comp", 0)}},
            {"key": "scale", "type": "ImageScale", "pos": (2, 4),
             "widgets": {"upscale_method": "bilinear", "width": 832, "height": 480,
                         "crop": "disabled"},
             "wire": {"image": ("frames", 0)}},
            # The source latent is the starting point: FlowRVS keeps
            # IMG_TO_IMG_FLOW, whose noise scaling returns the latent unchanged.
            {"key": "enc", "type": "VAEEncode", "pos": (3, 4),
             "wire": {"pixels": ("scale", 0), "vae": ("vae", 0)}},
            # One loader for both ends: the mask VAE's encoder half is the stock
            # Wan encoder and only its decoder head changed, to one channel.
            {"key": "vae", "type": "VAELoader", "pos": (3, 5),
             "widgets": {"vae_name": "wan21_flow_rvs_mask_vae_bf16.safetensors"}},
            # One step of euler on simple puts the DiT at timestep 999 on the
            # eps-style table this family keeps; a hand-written 1.0 sigma would
            # land on 354 instead.
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 1, "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("enc", 0)}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            # The decode returns one channel, which Save Image cannot write.
            # Mask and back is the official bridge to three channels.
            {"key": "tomask", "type": "ImageToMask", "pos": (4, 2),
             "widgets": {"channel": "red"},
             "wire": {"image": ("dec", 0)}},
            {"key": "toimage", "type": "MaskToImage", "pos": (5, 2),
             "wire": {"mask": ("tomask", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan_flowrvs"},
             "wire": {"images": ("toimage", 0)}},
        ],
    },
    "dgx-monarch-wan-bernini": {
        "title": "Wan Bernini-R text to video (high and low experts)",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan Bernini-R text to video at 832x480, 81 frames (about 5 s at 16 fps), "
             "40 steps at cfg 5.0 split between the two experts: the high-noise "
             "checkpoint runs steps 0 to 20 and hands the part-denoised latent to the "
             "low-noise one for steps 20 to 40, and each stage starts its own "
             "res_multistep solver history. Bernini-R is a renderer fine tune of Wan "
             "2.2 and loads through the stock Wan path, so auto splits the token "
             "sequence, not the CFG batch; the Bernini Conditioning node needs "
             "ComfyUI f8e51b67 or newer. The two experts share one model slot on the "
             "workers, so a render loads the high checkpoint and then swaps it for "
             "the low one, and each checkpoint runs its own first-render identity "
             "gate. With nothing wired into it, Bernini Conditioning is text to "
             "video; wiring a source video or reference images into it asks for the "
             "in-context conditioning path, which no recorded render has measured. "
             "The reference_images socket is a growable group with a minimum of "
             "none, so this graph ships it collapsed: for the r2v and rv2v tasks, "
             "grow it on the node and wire your own images. Sampling shift is Wan's "
             "own 8.0, which the public Comfy Bernini graph inherits by setting no "
             "shift node. The vendor demo runs flow shift 5.0 with UniPC and APG "
             "guidance, and this graph reproduces neither, so change the widget "
             "only to try that shift. The negative prompt is an English rendering "
             "of the vendor default, not a string the vendor published. For the "
             "six-step preview recipe, add Load LoRA (DGX Monarch) with "
             "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors from "
             "models/loras at 3.0 on the high expert and 1.5 on the low one, set "
             "both stages to 6 steps split at 3, and drop cfg to 1.0. The "
             "int8_convrot pair in the same repository has the same shape and loads "
             "the same way, with no fidelity claim here. Tests "
             "cover the FP8-scaled high/low auto/Ulysses text-to-video graph: cold "
             "and warm 81-frame renders completed, and the decoded one-step probe "
             "matched its local reference exactly. It does not cover in-context "
             "conditioning, other quantizations, FSDP, LoRA or continuous solver "
             "history. "
             + FRAMES + " " + GATE + " Expects "
             "wan2.2_bernini_r_high_noise_fp8_scaled.safetensors plus "
             "wan2.2_bernini_r_low_noise_fp8_scaled.safetensors in "
             "models/diffusion_models, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, and wan_2.1_vae.safetensors in models/vae. " +
             DERIVED},
            {"key": "high", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "wan2.2_bernini_r_high_noise_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "low", "type": "DGXMonarchUNETLoader", "pos": (1, 1),
             "widgets": {"unet_name":
                         "wan2.2_bernini_r_low_noise_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # One shift node per expert: the patch applies to the resident model,
            # and the two stages are two separate models on the mesh.
            {"key": "shift_high", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 8.0}, "wire": {"model": ("high", 0)}},
            {"key": "shift_low", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 1),
             "widgets": {"shift": 8.0}, "wire": {"model": ("low", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "Day time, side lighting, medium shot, center "
                                 "composition. A large, fluffy white polar bear sits "
                                 "upright on a snowbank, holding a brown wooden "
                                 "acoustic guitar. The bear's thick furry right paw "
                                 "continuously strums the metal strings up and down, "
                                 "while its heavy body sways side to side. The bear's "
                                 "dark black nose twitches, and its mouth is slightly "
                                 "open as it moves its head. Behind the polar bear, "
                                 "white snowflakes gently drift downward across a vast "
                                 "icy landscape under a bright, deep blue sky. "
                                 "Sunlight casts crisp shadows on the snowy ground, "
                                 "illuminating the bear's thick fur and the polished "
                                 "surface of the guitar."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "oversaturated, overexposed, static, blurry details, "
                                 "subtitles, style, artwork, painting, still image, "
                                 "overall grey, worst quality, low quality, JPEG "
                                 "compression artifacts, ugly, mutilated, extra "
                                 "fingers, badly drawn hands, badly drawn face, "
                                 "deformed, disfigured, malformed limbs, fused "
                                 "fingers, motionless frame, cluttered background, "
                                 "three legs, crowd in the background, walking "
                                 "backwards"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            # With no media wired this node is a pass-through plus an empty
            # latent of the same shape Empty Hunyuan Latent Video builds, and it
            # sets no conditioning keys at all. It is here because it names the
            # family and carries the media sockets for the other tasks.
            {"key": "cond", "type": "BerniniConditioning", "pos": (2, 2),
             "widgets": {"width": 832, "height": 480, "length": 81, "batch_size": 1,
                         "ref_max_size": 848},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0)}},
            # 40 steps of the `simple` schedule cut at 20 is the sigma pair the
            # public graph builds with Basic Scheduler plus Split Sigmas. Basic
            # Scheduler (DGX Monarch) refuses under topology auto, so the two
            # stages are KSampler Advanced instead.
            {"key": "ks_high", "type": "DGXMonarchKSamplerAdvanced", "pos": (3, 0),
             "widgets": {"add_noise": "enable", "noise_seed": 42, "steps": 40,
                         "cfg": 5.0, "sampler_name": "res_multistep",
                         "scheduler": "simple", "start_at_step": 0, "end_at_step": 20,
                         "return_with_leftover_noise": "enable"},
             "wire": {"model": ("shift_high", 0), "positive": ("cond", 0),
                      "negative": ("cond", 1), "latent_image": ("cond", 2)}},
            # add_noise stays disabled here: the latent arrives from the high
            # stage still carrying its leftover noise.
            {"key": "ks_low", "type": "DGXMonarchKSamplerAdvanced", "pos": (3, 1),
             "widgets": {"add_noise": "disable", "noise_seed": 0, "steps": 40,
                         "cfg": 5.0, "sampler_name": "res_multistep",
                         "scheduler": "simple", "start_at_step": 20,
                         "end_at_step": 10000,
                         "return_with_leftover_noise": "disable"},
             "wire": {"model": ("shift_low", 0), "positive": ("cond", 0),
                      "negative": ("cond", 1), "latent_image": ("ks_high", 0)}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks_low", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan_bernini"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-wan-scail": {
        "title": "Wan SCAIL Preview character animation",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan SCAIL Preview animates one reference character image at 512x896, 81 "
             "frames (about 5 s at 16 fps), 40 steps at cfg 5.0 with uni_pc and an SD3 "
             "sampling shift of 3.0, the vendor's own defaults. This recipe rendered, "
             "but its reference failed for capacity; that comparison remains CHECK. "
             "Hardware fidelity covers only synthetic 256x256, five-frame tests at "
             "one and four steps, using Ulysses and Torch Flash with the BF16 "
             "checkpoint executing in FP16. It does not qualify this default recipe, "
             "pose input, LoRA, ring or FSDP. See docs/VALIDATION.md for details. "
             "The driver prices the render before "
             "dispatch and can still refuse it (docs/TROUBLESHOOTING.md #47). Run the "
             "small probe from #47 first, 384x384 at length 17 with this one "
             "reference image, no pose video, and auto_gate off on the Init node, "
             "then return to this geometry; width and height must stay divisible by "
             "32, and length must stay on the 4k+1 grid. SCAIL normally takes its "
             "motion from a rendered 3D-consistent pose video, and no stock node "
             "draws one, so pose_video is left unwired here and the clip animates "
             "from the reference image and the prompt alone. To drive a specific "
             "performance, render a pose video with the SCAIL pose pack at half "
             "this resolution and wire it in. Point Load Image at your own "
             "character; the shipped filename is a placeholder. The model was "
             "trained on long detailed prompts, so keep the description of subject, "
             "motion and scene. Expects "
             "Wan21-14B-SCAIL-preview_comfy_bf16.safetensors in "
             "models/diffusion_models, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, clip_vision_h.safetensors in models/clip_vision, "
             "and wan_2.1_vae.safetensors in models/vae; "
             "Wan21-14B-SCAIL-preview_fp8_scaled_mixed.safetensors halves the weights "
             "resident on each rank if bf16 does not fit, but it gives up the FSDP "
             "capacity route, which takes bf16 only. 40 steps at cfg 5.0 on a 14B video "
             "model is a long first render: the lightx2v rank64 step and cfg distill "
             "LoRA runs it in 6 steps at cfg 1.0 with shift 7.0 and dpmpp_2m_sde, "
             "which this graph does not wire because the distilled recipe is not the "
             "contract this row was built against. " + FRAMES + " " + GATE + " "
             + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "Wan21-14B-SCAIL-preview_comfy_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # The vendor's default sample_shift is 3.0. It goes through the DGX
            # Monarch node: the stock patch node cannot take a mesh model.
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 3.0}, "wire": {"model": ("unet", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A dancer with long braided hair moves along a "
                                 "windswept rocky shoreline at golden hour, wearing a "
                                 "flowing teal jacket over dark trousers and worn "
                                 "leather boots. She turns through a full spin, lifts "
                                 "both arms overhead, then steps forward and leans "
                                 "into the sea breeze as her jacket and hair trail "
                                 "the motion. Waves break on the rocks behind her, "
                                 "gulls cross the sky, and low warm sunlight rims her "
                                 "shoulders while the camera holds a steady "
                                 "medium-wide shot."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low quality, static, watermark, text, "
                                 "distorted"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "image", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "example.png", "upload": "image"}},
            {"key": "cvload", "type": "CLIPVisionLoader", "pos": (0, 4),
             "widgets": {"clip_name": "clip_vision_h.safetensors"}},
            # crop stays none: the model is trained with a stretch resize to the
            # target aspect, so a centre crop throws away the edges it expects.
            {"key": "cvenc", "type": "CLIPVisionEncode", "pos": (1, 4),
             "widgets": {"crop": "none"},
             "wire": {"clip_vision": ("cvload", 0), "image": ("image", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            # This node builds the empty video latent itself and writes the
            # reference latent, the CLIP vision features and the ref mask flag
            # onto both conditionings, so the sampler must read positive and
            # negative from here, not straight from the text encodes.
            # pose_video, pose_video_mask, reference_image_mask and
            # previous_frames stay unwired: the last three are SCAIL-2 only.
            {"key": "scail", "type": "WanSCAILToVideo", "pos": (2, 3),
             "widgets": {"width": 512, "height": 896, "length": 81,
                         "batch_size": 1, "pose_strength": 1.0, "pose_start": 0.0,
                         "pose_end": 1.0, "video_frame_offset": 0,
                         "previous_frame_count": 5, "replacement_mode": False},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0), "reference_image": ("image", 0),
                      "clip_vision_output": ("cvenc", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 40, "cfg": 5.0, "sampler_name": "uni_pc",
                         "scheduler": "simple"},
             "wire": {"model": ("shift", 0), "positive": ("scail", 0),
                      "negative": ("scail", 1), "latent_image": ("scail", 2)}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan_scail"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-wan-scail2": {
        "title": "Wan SCAIL-2 character replacement",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Wan SCAIL-2 puts your character into your driving video at 896x512, 81 "
             "frames (about 5 s at 16 fps), 40 steps at cfg 5.0 with euler on the "
             "simple schedule, an SD3 sampling shift of 5.0 and the DPO LoRA at "
             "strength 1.0. That is the official Comfy base recipe with turbo off, "
             "and it is the contract docs/VALIDATION.md records. It takes four media "
             "inputs, all yours: no ComfyUI install ships any of them, so all four "
             "shipped filenames are placeholders. One Load Video takes the driving "
             "clip and the other takes its colored per-identity mask; one Load Image "
             "takes the reference character and the other takes that character's "
             "mask. Both modes require the masks. This graph ships Replacement Mode, "
             "which wants a white background on the driving mask and a black "
             "background on the reference mask; Animation Mode (replacement_mode "
             "false) wants the reverse, and without a correct mask Animation behaves "
             "like Replacement. To have the masks drawn for you, add the stock SAM3 "
             "nodes: Load Checkpoint on sam3.1_multiplex_fp16.safetensors, a CLIP Text "
             "Encode naming the tracking target (the official graph writes `human`, "
             "which is a tracking target, not the render prompt), one SAM3 Video "
             "Track per media input, then Create SCAIL-2 Colored Mask, whose own "
             "replacement_mode must match the one here. That route needs its own "
             "SAM3 checkpoint, so this graph leaves it out and starts from mask media "
             "you already have. Wan SCAIL-2 resizes each input itself, the pose pair "
             "to half the render size and the reference pair to full size, so the "
             "four files need not match in resolution. Frame counts do matter: the "
             "driving clip and its mask are trimmed together to the shorter of the "
             "two. Width and height must both divide by 32, and length sits on the "
             "4k+1 grid. The graph renders one 81-frame chunk; longer video is "
             "manual, ceil(total_frames / 76) Queue presses, each wiring "
             "video_frame_offset from the previous chunk's fourth output and "
             "previous_frames from the previous chunk's decoded frames. Nothing here "
             "queues its own segments. The negative is empty, as the official "
             "SCAIL-2 template ships it, instead of the long Wan 2.1 house negative "
             "other graphs here use. 40 steps at cfg 5.0 on a 14B video model is a "
             "long first render, and the driver prices the activation footprint "
             "before dispatch and can refuse it (docs/TROUBLESHOOTING.md #47): run "
             "the small probe from #47 first, 384x384 at length 17 with auto_gate "
             "off on the Init node, then return to this geometry. The official fast "
             "preset (lightx2v rank64 at 0.8, six steps, cfg 1.0) is left out, "
             "because a distilled cfg 1.0 path is not the sampler contract this row "
             "was measured against. Tests cover FP8-scaled "
             "synthetic-conditioning auto/Ulysses: cold and warm renders completed, "
             "and the decoded 81-frame one-step probe matched its local reference "
             "exactly. Arbitrary driving media, Turbo, FSDP and unlisted LoRA "
             "settings have no passing reference comparison. " + FRAMES + " " + GATE +
             " Expects wan2.1_14B_SCAIL_2_fp8_scaled.safetensors in "
             "models/diffusion_models, wan2.1_SCAIL_2_DPO_lora_bf16.safetensors in "
             "models/loras, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, clip_vision_h.safetensors in models/clip_vision, "
             "and wan_2.1_vae.safetensors in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # The DPO LoRA is part of the base recipe, not an option: the row was
            # measured with it at strength 1.0 and turbo off.
            {"key": "lora", "type": "DGXMonarchLoraLoader", "pos": (2, 0),
             "widgets": {"lora_name": "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
                         "strength_model": 1.0},
             "wire": {"model": ("unet", 0)}},
            # The shift goes through the DGX Monarch node: the stock one cannot take
            # a mesh model.
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (3, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("lora", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A young woman with dark hair tied in a neat high "
                                 "bun, with a few loose strands framing her face, is "
                                 "dancing outdoors on a sunny coastal hillside. She "
                                 "has a normal-sized head and a slim face, with no "
                                 "hat, no headwear, and no oversized hair volume. She "
                                 "wears a fitted black long-sleeve crop top with a "
                                 "shoulder cutout, extremely baggy black cargo pants "
                                 "with straps and pockets, and chunky black combat "
                                 "boots. She performs energetic dance moves with one "
                                 "leg lifted and arms extended, moving naturally in "
                                 "front of a large tree, a small white stone house "
                                 "with a terracotta roof, and a bright blue sea under "
                                 "a clear sky with light clouds."},
             "wire": {"clip": ("clip", 0)}},
            # Empty, like the official template: this family ships no house negative.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": ""},
             "wire": {"clip": ("clip", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 6),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            {"key": "video", "type": "LoadVideo", "pos": (0, 4),
             "widgets": {"file": "driving_video.mp4", "upload": "image"}},
            {"key": "comp", "type": "GetVideoComponents", "pos": (1, 4),
             "wire": {"video": ("video", 0)}},
            # The mask video is a second media file, not a derivative of the first:
            # Create SCAIL-2 Colored Mask needs SAM3 track data on its driving side
            # and cannot take a plain mask there, so a graph without SAM3 takes the
            # colored mask as media.
            {"key": "maskvideo", "type": "LoadVideo", "pos": (0, 5),
             "widgets": {"file": "driving_video_mask.mp4", "upload": "image"}},
            {"key": "maskcomp", "type": "GetVideoComponents", "pos": (1, 5),
             "wire": {"video": ("maskvideo", 0)}},
            {"key": "image", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "reference_character.png", "upload": "image"}},
            {"key": "maskimage", "type": "LoadImage", "pos": (0, 7),
             "widgets": {"image": "reference_character_mask.png", "upload": "image"}},
            {"key": "cvload", "type": "CLIPVisionLoader", "pos": (1, 6),
             "widgets": {"clip_name": "clip_vision_h.safetensors"}},
            # crop stays none, as the official template stores it (reason at the
            # SCAIL Preview graph's CLIP Vision Encode).
            {"key": "cvenc", "type": "CLIPVisionEncode", "pos": (2, 6),
             "widgets": {"crop": "none"},
             "wire": {"clip_vision": ("cvload", 0), "image": ("image", 0)}},
            # Both mask sockets are IMAGE, not MASK: the colored mask carries one
            # palette color per identity, which a single-channel mask cannot.
            # previous_frames stays unwired because this is the first chunk.
            {"key": "scail", "type": "WanSCAILToVideo", "pos": (2, 3),
             "widgets": {"width": 896, "height": 512, "length": 81,
                         "batch_size": 1, "pose_strength": 1.0, "pose_start": 0.0,
                         "pose_end": 1.0, "video_frame_offset": 0,
                         "previous_frame_count": 5, "replacement_mode": True},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0), "pose_video": ("comp", 0),
                      "pose_video_mask": ("maskcomp", 0),
                      "reference_image": ("image", 0),
                      "reference_image_mask": ("maskimage", 0),
                      "clip_vision_output": ("cvenc", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (4, 1),
             "widgets": {"steps": 40, "cfg": 5.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0},
             "wire": {"model": ("shift", 0), "positive": ("scail", 0),
                      "negative": ("scail", 1), "latent_image": ("scail", 2)}},
            {"key": "dec", "type": "VAEDecode", "pos": (5, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (6, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wan_scail2"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-wandancer": {
        "title": "WanDancer music to dance, global to local",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "WanDancer music to dance at 480x832, 149 frames a segment: the global "
             "checkpoint plans the whole track at a low frame rate, that plan is cut "
             "to one image per latent frame to make the keyframes, and the local "
             "checkpoint refines the first five seconds at 30 fps, 25 steps then 24 "
             "steps, both at cfg 5.0. Point Load Image at one reference dancer and "
             "Load Audio at a music track of about 25 seconds, the length the "
             "official graph trims to; the shipped names are the official demo files "
             "blue_dancer.png and wan_dancer_reference_audio.mp3. Both stages need "
             "the reference image: without start_image and CLIP vision every "
             "topology returns non-finite latents, which stock ComfyUI writes out as "
             "black frames. Track length sets the rest: the global pass reads its "
             "frame rate as 149 frames over the track, so a track near five seconds "
             "lands on 30 fps and runs the local head on the global checkpoint, "
             "while a very long track leaves almost no keyframes in each segment. "
             "num_segments counts five-second steps of finished video, so raise it "
             "only as far as your track reaches. Only the global expert takes a "
             "Model Sampling SD3 node, because the official graph patches no model "
             "sampling onto the local checkpoint; adding a second one changes the "
             "local stage's math. Reference Latent typed-refuses on this family, FP8 "
             "with FSDP is out of scope, and the official Skip Layer Guidance patch "
             "cannot be wired here because a mesh model is not a stock MODEL. For "
             "the official fast path, put "
             "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors in "
             "models/loras, add a Load LoRA (DGX Monarch) after each Load Diffusion "
             "Model (DGX Monarch) at strength 3.0 global and 1.03 local, and drop "
             "both stages to 6 steps at cfg 1.0; nothing here has run that route. " +
             FRAMES + " " + GATE + " Expects "
             "wan2.2_dancer_14b_global_fp8_scaled.safetensors and "
             "wan2.2_dancer_14b_local_fp8_scaled.safetensors in "
             "models/diffusion_models, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
             "models/text_encoders, clip_vision_h.safetensors in models/clip_vision, "
             "and wan_2.1_vae.safetensors in models/vae. " + DERIVED},
            {"key": "gunet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "wan2.2_dancer_14b_global_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # Only the global branch takes a shift: the official graph applies no
            # model sampling on the local checkpoint.
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("gunet", 0)}},
            # Two checkpoints, not one. The release ships separate global and
            # local experts, and a graph that samples one of them twice is wrong.
            {"key": "lunet", "type": "DGXMonarchUNETLoader", "pos": (1, 1),
             "widgets": {"unet_name": "wan2.2_dancer_14b_local_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "wan"}},
            # The vendor ships its dance-genre prompts in Chinese, one file per
            # genre, and the official template's strings are Chinese too.
            {"key": "posg", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "一个人正在跳舞，舞蹈种类是拉丁舞"},  # noqa: RUF001
             "wire": {"clip": ("clip", 0)}},
            # The local prompt is the global one plus the clarity and amplitude
            # clause the official graph appends. That graph also pastes the audio
            # encoder's frame-rate string into the global prompt; the wire for it
            # ends in a text widget, a shape this generator does not serialize,
            # and the frame rate itself reaches the model through the audio
            # encoder output either way.
            {"key": "posl", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text":
                         "一个人正在跳舞，舞蹈种类是拉丁舞,图像清晰程度高,人物动作幅度中等"},  # noqa: RUF001
             "wire": {"clip": ("clip", 0)}},
            # Both stages run at cfg 5.0, so both take the real negative. The
            # official graph zeroes its local negative only because that branch
            # is a distilled cfg 1.0 pass where nothing reads it.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 4),
             "widgets": {"text":
                         "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"  # noqa: RUF001
                         "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"  # noqa: RUF001
                         "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"  # noqa: RUF001
                         "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"  # noqa: RUF001
                         "三条腿，背景人很多，倒着走"},  # noqa: RUF001
             "wire": {"clip": ("clip", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 3),
             "widgets": {"vae_name": "wan_2.1_vae.safetensors"}},
            {"key": "cvload", "type": "CLIPVisionLoader", "pos": (0, 4),
             "widgets": {"clip_name": "clip_vision_h.safetensors"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 5),
             "widgets": {"image": "blue_dancer.png", "upload": "image"}},
            {"key": "audio", "type": "LoadAudio", "pos": (0, 6),
             "widgets": {"audio": "wan_dancer_reference_audio.mp3",
                         "audioUI": None, "upload": None}},
            {"key": "cvref", "type": "CLIPVisionEncode", "pos": (1, 5),
             "widgets": {"crop": "center"},
             "wire": {"clip_vision": ("cvload", 0), "image": ("image", 0)}},
            {"key": "genc", "type": "WanDancerEncodeAudio", "pos": (1, 6),
             "widgets": {"video_frames": 149, "audio_inject_scale": 1.0},
             "wire": {"audio": ("audio", 0)}},
            # Global stage: the reference image is both the first frame and the
            # reference embedding, and mask stays unconnected so the whole span
            # generates.
            {"key": "gvid", "type": "WanDancerVideo", "pos": (2, 2),
             "widgets": {"width": 480, "height": 832, "length": 149},
             "wire": {"positive": ("posg", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0), "clip_vision_output": ("cvref", 0),
                      "clip_vision_output_ref": ("cvref", 0),
                      "start_image": ("image", 0),
                      "audio_encoder_output": ("genc", 0)}},
            {"key": "gks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 25, "cfg": 5.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0},
             "wire": {"model": ("shift", 0), "positive": ("gvid", 0),
                      "negative": ("gvid", 1), "latent_image": ("gvid", 2)}},
            # One image per latent frame, not per pixel frame. The bridge wants a
            # sparse plan to hang the 30 fps segment on; decoding the whole global
            # latent hands the pad node 149 keyframes instead of 38 and pins one
            # output frame in five, which locks the local expert to an upsampled
            # low-frame-rate plan.
            {"key": "gcut", "type": "LatentCutToBatch", "pos": (4, 1),
             "widgets": {"dim": "t", "slice_size": 1},
             "wire": {"samples": ("gks", 0)}},
            {"key": "gdec", "type": "VAEDecode", "pos": (5, 1),
             "wire": {"samples": ("gcut", 0), "vae": ("vae", 0)}},
            # The pad node has three LIST outputs. ComfyUI iterates everything
            # downstream once per emitted segment; monarch adds no loop of its own.
            {"key": "pad", "type": "WanDancerPadKeyframesList", "pos": (6, 1),
             "widgets": {"segment_length": 149, "num_segments": 1},
             "wire": {"images": ("gdec", 0), "audio": ("audio", 0)}},
            {"key": "lenc", "type": "WanDancerEncodeAudio", "pos": (6, 3),
             "widgets": {"video_frames": 149, "audio_inject_scale": 1.0},
             "wire": {"audio": ("pad", 2)}},
            {"key": "first", "type": "ImageFromBatch", "pos": (6, 4),
             "widgets": {"batch_index": 0, "length": 1},
             "wire": {"image": ("pad", 0)}},
            {"key": "cvkey", "type": "CLIPVisionEncode", "pos": (7, 4),
             "widgets": {"crop": "center"},
             "wire": {"clip_vision": ("cvload", 0), "image": ("first", 0)}},
            # Local stage: clip_vision_output is this segment's first keyframe,
            # clip_vision_output_ref stays the reference dancer.
            {"key": "lvid", "type": "WanDancerVideo", "pos": (7, 2),
             "widgets": {"width": 480, "height": 832, "length": 149},
             "wire": {"positive": ("posl", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0), "clip_vision_output": ("cvkey", 0),
                      "clip_vision_output_ref": ("cvref", 0),
                      "start_image": ("pad", 0), "mask": ("pad", 1),
                      "audio_encoder_output": ("lenc", 0)}},
            {"key": "lks", "type": "DGXMonarchKSampler", "pos": (8, 1),
             "widgets": {"steps": 24, "cfg": 5.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0},
             "wire": {"model": ("lunet", 0), "positive": ("lvid", 0),
                      "negative": ("lvid", 1), "latent_image": ("lvid", 2)}},
            {"key": "ldec", "type": "VAEDecode", "pos": (9, 1),
             "wire": {"samples": ("lks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (10, 1),
             "widgets": {"filename_prefix": "dgx_monarch_wandancer"},
             "wire": {"images": ("ldec", 0)}},
        ],
    },
    "dgx-monarch-ltx-t2v": {
        "title": "LTX 2.3 text to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LTX 2.3 text to video at 1280x704, 121 frames (about 5 s at 24 fps), 8 "
             "steps on the distilled checkpoint at cfg 1.0. The official graph asks "
             "for 1280x720 and lands on the same latent, because the latent math "
             "floors the height to a multiple of 32. Video sequences are long, "
             "which is what sequence-parallel is for, so auto picks uly2 on a "
             "configured pair. The graph uses the custom-sampler route because the "
             "distilled checkpoint was tuned on a fixed sigma schedule: the nine "
             "values are typed into Manual Sigmas instead of computed by a "
             "scheduler node, and Sampler Custom (DGX Monarch) takes them without "
             "needing a mesh model. The two text-encoder files load through one "
             "Dual CLIP Loader set to type ltxv. Large video latents can come back "
             "over RDMA; that path is off by default and its native leg is NOT RUN / "
             "HOLD (DESIGN.md 5.6). " + FRAMES + " " + GATE + " Expects "
             "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors in "
             "models/diffusion_models, gemma_3_12B_it.safetensors plus "
             "ltx-2.3_text_projection_bf16.safetensors in models/text_encoders, and "
             "LTX23_video_vae_bf16.safetensors in models/vae. The split layout is "
             "needed because a transformer-only checkpoint carries no VAE, so the "
             "VAE and the decode tail come from the stock node definitions."},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "gemma_3_12B_it.safetensors",
                         "clip_name2": "ltx-2.3_text_projection_bf16.safetensors",
                         "type": "ltxv"}},
            # The 2.3 prompting guidance asks for a long chronological prompt that
            # names the single camera move and describes the soundtrack.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "One unbroken take. The camera pushes in slowly and "
                                 "steadily, holding level, then settles into a medium "
                                 "framing and holds there. Light moves across the "
                                 "surfaces as the air stirs, loose material drifts "
                                 "and settles, and the background stays soft. Nothing "
                                 "cuts and the camera never pulls back. Audio: a "
                                 "quiet room tone under moving air, small close "
                                 "sounds near the lens, one distant bell."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "pc game, console game, video game, cartoon, "
                                 "childish, ugly"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "ltxc", "type": "LTXVConditioning", "pos": (2, 2),
             "widgets": {"frame_rate": 24.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0)}},
            {"key": "latent", "type": "EmptyLTXVLatentVideo", "pos": (2, 3),
             "widgets": {"width": 1280, "height": 704, "length": 121,
                         "batch_size": 1}},
            # The official 2.3 graphs, all but the two IC-LoRA ones, hand the sampler
            # this fixed nine-value list.
            # The shift-plus-terminal curve the LTXV Scheduler builds is the LTXV
            # 0.9-era recipe, not what the distilled 2.3 checkpoint was tuned on,
            # so the schedule is typed, not computed, and needs no latent wire.
            {"key": "sched", "type": "ManualSigmas", "pos": (2, 4),
             "widgets": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, "
                                   "0.725, 0.421875, 0.0"}},
            {"key": "sampler", "type": "KSamplerSelect", "pos": (1, 4),
             "widgets": {"sampler_name": "euler"}},
            {"key": "noise", "type": "RandomNoise", "pos": (1, 5),
             "widgets": {"noise_seed": 42}},
            {"key": "guider", "type": "DGXMonarchCFGGuider", "pos": (3, 0),
             "widgets": {"cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("ltxc", 0),
                      "negative": ("ltxc", 1)}},
            {"key": "sc", "type": "DGXMonarchSamplerCustom", "pos": (3, 1),
             "wire": {"noise": ("noise", 0), "guider": ("guider", 0),
                      "sampler": ("sampler", 0), "sigmas": ("sched", 0),
                      "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("sc", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ltx_t2v"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-ltx-i2v-av": {
        "title": "LTX 2.3 image to video with sound",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LTX 2.3 image to video with sound at 1280x704, 121 frames (about 5 s at 24 "
             "fps): one packed video and audio latent, denoised once, then decoded "
             "twice, the frames through the video VAE and the soundtrack through the "
             "audio VAE. This is the concat route: LTXV Image To Video encodes your "
             "first frame into the head of an otherwise empty video latent and "
             "returns a noise mask that holds it still while the rest denoises. It "
             "sets no conditioning keys, so the prompt wiring matches text to video. "
             "Point Load Image at your own first frame; the shipped filename is a "
             "placeholder. The distilled checkpoint runs 8 steps at cfg 1.0 on a "
             "fixed sigma schedule, typed into Manual Sigmas instead of computed by "
             "a scheduler node, and sampled with euler. This graph runs one pass; "
             "the official graph upsamples the latent and runs the 22B DiT a second "
             "time. The canvas gives 14080 video rows and 126 audio rows, both even, "
             "so neither stream pads at world 2. Packed data-parallel and FP8 FSDP "
             "each refuse with a typed refusal, and cfg-parallel is unsupported "
             "here, so cfg stays 1.0. Describe the soundtrack after Audio: in the "
             "prompt, and quote a line of dialogue to make the model speak it. "
             + FRAMES + " " + GATE +
             " Expects ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors in "
             "models/diffusion_models, gemma_3_12B_it.safetensors and "
             "ltx-2.3_text_projection_bf16.safetensors in models/text_encoders through "
             "one Dual CLIP Loader at type ltxv, and LTX23_video_vae_bf16.safetensors "
             "plus LTX23_audio_vae_bf16.safetensors in models/vae. Swap clip_name1 for "
             "gemma_3_12B_it_fp8_scaled.safetensors if the driver host is short of "
             "memory. "
             + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # LTX 2.3 keeps the text projection in a second file, so this is a dual
            # loader where the LTX 2.5 templates need only one.
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "gemma_3_12B_it.safetensors",
                         "clip_name2": "ltx-2.3_text_projection_bf16.safetensors",
                         "type": "ltxv"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "One unbroken take. The camera pushes in slowly and "
                                 "steadily, holding level, then settles into a medium "
                                 "framing and holds there. Light moves across the "
                                 "surfaces as the air stirs, loose material drifts and "
                                 "settles, and the background stays soft. Nothing cuts "
                                 "and the camera never pulls back. Audio: a quiet room "
                                 "tone under moving air, small close sounds near the "
                                 "lens, one distant bell."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "pc game, console game, video game, cartoon, childish, "
                                 "ugly"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "ltxc", "type": "LTXVConditioning", "pos": (2, 2),
             "widgets": {"frame_rate": 24.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0)}},
            {"key": "vvae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},
            {"key": "avae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "LTX23_audio_vae_bf16.safetensors"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "example.png", "upload": "image"}},
            # Width, height and length here replace the Empty LTXV Latent Video the
            # text to video graph uses: this node builds the latent itself. 1280x704
            # x 121 frames gives 14080 video rows and 126 audio rows, both even, so
            # neither stream pads at world 2 and no rank attends a synthetic row.
            {"key": "i2v", "type": "LTXVImgToVideo", "pos": (2, 3),
             "widgets": {"width": 1280, "height": 704, "length": 121,
                         "batch_size": 1, "strength": 1.0},
             "wire": {"positive": ("ltxc", 0), "negative": ("ltxc", 1),
                      "vae": ("vvae", 0), "image": ("image", 0)}},
            {"key": "alatent", "type": "LTXVEmptyLatentAudio", "pos": (1, 5),
             "widgets": {"frames_number": 121, "frame_rate": 24.0, "batch_size": 1},
             "wire": {"audio_vae": ("avae", 0)}},
            # Concat carries the video noise mask through as a packed pair, filling
            # the audio half with ones so the soundtrack denoises normally.
            {"key": "latent", "type": "LTXVConcatAVLatent", "pos": (2, 4),
             "wire": {"video_latent": ("i2v", 2),
                      "audio_latent": ("alatent", 0)}},
            {"key": "sigmas", "type": "ManualSigmas", "pos": (2, 5),
             "widgets": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, "
                                   "0.725, 0.421875, 0.0"}},
            # The released 2.3 recipe reaches this schedule through Sampler Euler
            # Ancestral at eta 0.0, which injects no ancestral noise and is plain
            # euler; asking KSamplerSelect for euler_ancestral would run comfy's
            # default eta 1.0 instead, which is a different sampler.
            {"key": "sampler", "type": "KSamplerSelect", "pos": (1, 6),
             "widgets": {"sampler_name": "euler"}},
            {"key": "noise", "type": "RandomNoise", "pos": (1, 7),
             "widgets": {"noise_seed": 42}},
            {"key": "guider", "type": "DGXMonarchCFGGuider", "pos": (3, 0),
             "widgets": {"cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("i2v", 0),
                      "negative": ("i2v", 1)}},
            {"key": "sc", "type": "DGXMonarchSamplerCustom", "pos": (3, 1),
             "wire": {"noise": ("noise", 0), "guider": ("guider", 0),
                      "sampler": ("sampler", 0), "sigmas": ("sigmas", 0),
                      "latent_image": ("latent", 0)}},
            {"key": "vdec", "type": "VAEDecodeTiled", "pos": (4, 1),
             "widgets": {"tile_size": 512, "overlap": 64, "temporal_size": 64,
                         "temporal_overlap": 16},
             "wire": {"samples": ("sc", 0), "vae": ("vvae", 0)}},
            {"key": "adec", "type": "LTXVAudioVAEDecode", "pos": (4, 2),
             "wire": {"samples": ("sc", 0), "audio_vae": ("avae", 0)}},
            {"key": "simg", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ltx_i2v_av"},
             "wire": {"images": ("vdec", 0)}},
            {"key": "saud", "type": "SaveAudioAdvanced", "pos": (5, 2),
             "widgets": {"filename_prefix": "audio/dgx_monarch_ltx_i2v_av",
                         "format": "flac"},
             "wire": {"audio": ("adec", 0)}},
        ],
    },
    "dgx-monarch-ltx25-t2v": {
        "title": "LTX 2.5 text to video with sound",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LTX 2.5 text to video with sound at 1280x704, 121 frames (about 5 s at "
             "24 fps). One packed video and audio latent is denoised once, then "
             "decoded twice: the frames through the video VAE and the soundtrack "
             "through the audio VAE. Both decode nodes take the packed latent and "
             "read their own half of it. The distilled checkpoint runs 8 steps at "
             "cfg 1.0 on a fixed sigma schedule, so the schedule is typed into "
             "Manual Sigmas instead of computed by a scheduler node; euler_ancestral "
             "is the sampler the released recipe uses. This is the single-stage "
             "shape: the official two-stage graph samples at half resolution and "
             "upsamples the video latent between two sampler passes, which runs the "
             "22B DiT twice. Video Decode Tiled is required at this size. The prompt "
             "enhancer is left out: it is a second Gemma text encoder resident "
             "beside the 15 GiB main one, so paste an enhanced prompt instead. The "
             "official template asks for 1280x720 and lands on this same latent, "
             "because Empty LTXV Latent Video builds height//32 rows and the model "
             "card wants both dimensions divisible by 32. Needs ComfyUI at "
             "bd34f338 or newer. The LTX 2.5 files sit behind a gated Hugging Face "
             "repo, so accept the license on Lightricks/LTX-2.5 first or the "
             "downloads fail. " + FRAMES + " " + GATE + " Expects "
             "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors in "
             "models/diffusion_models, "
             "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors in "
             "models/text_encoders, and ltx-2.5-video-vae-bf16.safetensors plus "
             "ltx-2.5-audio-vae-bf16.safetensors in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot"
                         ".safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # One file: LTX 2.5 folds the text projection into the encoder, so
            # this is a single loader where LTX 2.3 needed a dual one.
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name":
                         "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                         "type": "ltxv"}},
            # The vendor guide asks for one flowing paragraph covering shot, scene,
            # action, camera move and audio; the audio sits in the prose, not
            # behind a label, as in every official 2.5 prompt.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A low tracking shot glides along a wet cobblestone "
                                 "alley just after a rainstorm, blue evening light "
                                 "pooling in the puddles while warm lamplight spills "
                                 "from a corner bakery window. A tabby cat picks its "
                                 "way between the stones, paws lifting high, whiskers "
                                 "twitching at the steam curling off a grate. The "
                                 "camera pushes past the cat and rises into a shallow "
                                 "focus medium shot of the window, where a baker in a "
                                 "flour dusted apron slides a tray of rolls onto a "
                                 "rack, her breath fogging the cold glass for a "
                                 "moment before it clears. The push settles and holds "
                                 "on the glass. Water drips steadily from the awning, "
                                 "a distant tram bell rings twice, and the low hum of "
                                 "the ovens sits under the soft patter of the last "
                                 "rain."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "pc game, console game, video game, cartoon, "
                                 "childish, ugly"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "ltxc", "type": "LTXVConditioning", "pos": (2, 2),
             "widgets": {"frame_rate": 24.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0)}},
            {"key": "vvae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
            {"key": "avae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
            # 1280x704 x 121 frames: 14080 video rows and 126 audio rows, both
            # even, so neither stream pads at world 2.
            {"key": "vlatent", "type": "EmptyLTXVLatentVideo", "pos": (1, 4),
             "widgets": {"width": 1280, "height": 704, "length": 121, "batch_size": 1}},
            # frames_number and frame_rate describe the same clip as the video
            # latent and the conditioning; the audio length follows from them.
            {"key": "alatent", "type": "LTXVEmptyLatentAudio", "pos": (1, 5),
             "widgets": {"frames_number": 121, "frame_rate": 24.0, "batch_size": 1},
             "wire": {"audio_vae": ("avae", 0)}},
            {"key": "latent", "type": "LTXVConcatAVLatent", "pos": (2, 4),
             "wire": {"video_latent": ("vlatent", 0),
                      "audio_latent": ("alatent", 0)}},
            {"key": "sigmas", "type": "ManualSigmas", "pos": (2, 5),
             "widgets": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, "
                                   "0.725, 0.421875, 0.0"}},
            {"key": "sampler", "type": "KSamplerSelect", "pos": (1, 6),
             "widgets": {"sampler_name": "euler_ancestral"}},
            {"key": "noise", "type": "RandomNoise", "pos": (1, 7),
             "widgets": {"noise_seed": 42}},
            # The stock LTX 2.5 dual-CFG guider collapses to a single-scale
            # guider when its video and audio scales are equal, and the
            # distilled recipe sets both to 1.0.
            {"key": "guider", "type": "DGXMonarchCFGGuider", "pos": (3, 0),
             "widgets": {"cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("ltxc", 0),
                      "negative": ("ltxc", 1)}},
            {"key": "sc", "type": "DGXMonarchSamplerCustom", "pos": (3, 1),
             "wire": {"noise": ("noise", 0), "guider": ("guider", 0),
                      "sampler": ("sampler", 0), "sigmas": ("sigmas", 0),
                      "latent_image": ("latent", 0)}},
            {"key": "vdec", "type": "VAEDecodeTiled", "pos": (4, 1),
             "widgets": {"tile_size": 512, "overlap": 64, "temporal_size": 64,
                         "temporal_overlap": 16},
             "wire": {"samples": ("sc", 0), "vae": ("vvae", 0)}},
            {"key": "adec", "type": "LTXVAudioVAEDecode", "pos": (4, 2),
             "wire": {"samples": ("sc", 0), "audio_vae": ("avae", 0)}},
            {"key": "simg", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ltx25"},
             "wire": {"images": ("vdec", 0)}},
            {"key": "saud", "type": "SaveAudioAdvanced", "pos": (5, 2),
             "widgets": {"filename_prefix": "audio/dgx_monarch_ltx25",
                         "format": "flac"},
             "wire": {"audio": ("adec", 0)}},
        ],
    },
    "dgx-monarch-ltx25-i2v": {
        "title": "LTX 2.5 image to video with sound",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LTX 2.5 image to video with sound at 1280x704, 121 frames (about 5 s "
             "at 24 fps). This is the concat route: LTXV Image To Video encodes your "
             "first frame, writes it into an otherwise empty video latent and "
             "returns a per-frame noise mask that holds those frames still while the "
             "rest denoises. It sets no conditioning keys, so the prompt wiring "
             "matches text to video. Point Load Image at your own first frame; the "
             "shipped filename is a placeholder. strength 1.0 keeps the frame "
             "exactly, and lower values let the model redraw it. The geometry gives "
             "14080 video rows and 126 audio rows, both even, so neither stream pads "
             "at world 2. The official 2.5 image to video template takes the other "
             "route: LTXV Image To Video Inplace writes into an existing latent after "
             "the image passes through LTXV Preprocess at img_compression 18, and it "
             "runs strength 0.7 in the first of two stages and 1.0 in the second. "
             "This template takes the route the row proves, and one stage wants "
             "strength 1.0. " + FRAMES + " " + GATE + " Expects the same four files "
             "as the text to video template, from the same gated Hugging Face repo. "
             + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot"
                         ".safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name":
                         "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                         "type": "ltxv"}},
            # The official guidance for an image to video prompt: name the motion,
            # the camera move and the sounds that follow from the input image, and
            # do not re-describe what the frame already shows.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "Use the provided start image as the first frame. "
                                 "The camera pushes in slowly and steadily, holding "
                                 "the framing while the light shifts across the scene "
                                 "and dust turns in the air. Small movements build: "
                                 "fabric stirs, a shadow slides a little to the left, "
                                 "a reflection brightens and settles. The push in ends "
                                 "in a medium shot and holds. A low room tone runs "
                                 "under the shot, with distant traffic and the faint "
                                 "creak of a floorboard."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "pc game, console game, video game, cartoon, "
                                 "childish, ugly"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "ltxc", "type": "LTXVConditioning", "pos": (2, 2),
             "widgets": {"frame_rate": 24.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0)}},
            {"key": "vvae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
            {"key": "avae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "example.png", "upload": "image"}},
            # Width, height and length here replace the Empty LTXV Latent Video the
            # text to video graph uses: this node builds the latent itself.
            {"key": "i2v", "type": "LTXVImgToVideo", "pos": (2, 3),
             "widgets": {"width": 1280, "height": 704, "length": 121,
                         "batch_size": 1, "strength": 1.0},
             "wire": {"positive": ("ltxc", 0), "negative": ("ltxc", 1),
                      "vae": ("vvae", 0), "image": ("image", 0)}},
            {"key": "alatent", "type": "LTXVEmptyLatentAudio", "pos": (1, 5),
             "widgets": {"frames_number": 121, "frame_rate": 24.0, "batch_size": 1},
             "wire": {"audio_vae": ("avae", 0)}},
            # Concat keeps the video noise mask, as in the LTX 2.3 image to video graph.
            {"key": "latent", "type": "LTXVConcatAVLatent", "pos": (2, 4),
             "wire": {"video_latent": ("i2v", 2),
                      "audio_latent": ("alatent", 0)}},
            {"key": "sigmas", "type": "ManualSigmas", "pos": (2, 5),
             "widgets": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, "
                                   "0.725, 0.421875, 0.0"}},
            {"key": "sampler", "type": "KSamplerSelect", "pos": (1, 6),
             "widgets": {"sampler_name": "euler_ancestral"}},
            {"key": "noise", "type": "RandomNoise", "pos": (1, 7),
             "widgets": {"noise_seed": 42}},
            {"key": "guider", "type": "DGXMonarchCFGGuider", "pos": (3, 0),
             "widgets": {"cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("i2v", 0),
                      "negative": ("i2v", 1)}},
            {"key": "sc", "type": "DGXMonarchSamplerCustom", "pos": (3, 1),
             "wire": {"noise": ("noise", 0), "guider": ("guider", 0),
                      "sampler": ("sampler", 0), "sigmas": ("sigmas", 0),
                      "latent_image": ("latent", 0)}},
            {"key": "vdec", "type": "VAEDecodeTiled", "pos": (4, 1),
             "widgets": {"tile_size": 512, "overlap": 64, "temporal_size": 64,
                         "temporal_overlap": 16},
             "wire": {"samples": ("sc", 0), "vae": ("vvae", 0)}},
            {"key": "adec", "type": "LTXVAudioVAEDecode", "pos": (4, 2),
             "wire": {"samples": ("sc", 0), "audio_vae": ("avae", 0)}},
            {"key": "simg", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ltx25_i2v"},
             "wire": {"images": ("vdec", 0)}},
            {"key": "saud", "type": "SaveAudioAdvanced", "pos": (5, 2),
             "widgets": {"filename_prefix": "audio/dgx_monarch_ltx25_i2v",
                         "format": "flac"},
             "wire": {"audio": ("adec", 0)}},
        ],
    },
    "dgx-monarch-ltx25-i2v-guide": {
        "title": "LTX 2.5 image to video with sound (guide route)",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LTX 2.5 image to video at 1280x704, 121 frames, through the guide "
             "route. LTXV Add Guide appends your frame to the video latent as an "
             "extra latent frame and records its position on the conditioning, "
             "where the concat route writes the frame in place. The appended frame "
             "passes through the sampler as ordinary tokens and Crop Guides removes "
             "it afterwards, so keep Crop Guides in the graph or the decode returns "
             "one extra frame. Add Guide runs on the video latent before Concat AV "
             "Latent: it refuses a packed latent. Guide strength is not pinned to "
             "1.0 here. Any other value, or a connected attention_mask, asks for an "
             "additive self-attention bias, and Ulysses re-expresses that bias "
             "exactly: the shipped topology auto resolves to uly2 for LTX at every "
             "canvas, so an attenuated guide renders, while ring and hybrid refuse "
             "it before any pixels because the ring kernel takes no bias argument "
             "(docs/TROUBLESHOOTING.md #80). Topology 'single' and the cfg "
             "topologies run stock attention per rank and take it too. This graph "
             "ships strength 1.0, which runs on every topology; the released "
             "first-and-last-frame recipe uses 0.7, so a graph diffed against the "
             "stock template will not match at 1.0. Point Load Image at your own "
             "first frame. One guide frame gives 14960 video rows, still even. "
             + FRAMES + " " +
             GATE + " " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot"
                         ".safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name":
                         "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                         "type": "ltxv"}},
            # The prompt follows the official guidance given at the LTX 2.5 image to
            # video graph.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "Use the provided start image as the first frame. "
                                 "The camera pushes in slowly and steadily, holding "
                                 "the framing while the light shifts across the scene "
                                 "and dust turns in the air. Small movements build: "
                                 "fabric stirs, a shadow slides a little to the left, "
                                 "a reflection brightens and settles. The push in ends "
                                 "in a medium shot and holds. A low room tone runs "
                                 "under the shot, with distant traffic and the faint "
                                 "creak of a floorboard."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "pc game, console game, video game, cartoon, "
                                 "childish, ugly"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "ltxc", "type": "LTXVConditioning", "pos": (2, 2),
             "widgets": {"frame_rate": 24.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0)}},
            {"key": "vvae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
            {"key": "avae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "example.png", "upload": "image"}},
            {"key": "vlatent", "type": "EmptyLTXVLatentVideo", "pos": (1, 4),
             "widgets": {"width": 1280, "height": 704, "length": 121, "batch_size": 1}},
            {"key": "guide", "type": "LTXVAddGuide", "pos": (2, 3),
             "widgets": {"frame_idx": 0, "strength": 1.0},
             "wire": {"positive": ("ltxc", 0), "negative": ("ltxc", 1),
                      "vae": ("vvae", 0), "latent": ("vlatent", 0),
                      "image": ("image", 0)}},
            {"key": "alatent", "type": "LTXVEmptyLatentAudio", "pos": (1, 5),
             "widgets": {"frames_number": 121, "frame_rate": 24.0, "batch_size": 1},
             "wire": {"audio_vae": ("avae", 0)}},
            {"key": "latent", "type": "LTXVConcatAVLatent", "pos": (2, 4),
             "wire": {"video_latent": ("guide", 2),
                      "audio_latent": ("alatent", 0)}},
            {"key": "sigmas", "type": "ManualSigmas", "pos": (2, 5),
             "widgets": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, "
                                   "0.725, 0.421875, 0.0"}},
            {"key": "sampler", "type": "KSamplerSelect", "pos": (1, 6),
             "widgets": {"sampler_name": "euler_ancestral"}},
            {"key": "noise", "type": "RandomNoise", "pos": (1, 7),
             "widgets": {"noise_seed": 42}},
            {"key": "guider", "type": "DGXMonarchCFGGuider", "pos": (3, 0),
             "widgets": {"cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("guide", 0),
                      "negative": ("guide", 1)}},
            {"key": "sc", "type": "DGXMonarchSamplerCustom", "pos": (3, 1),
             "wire": {"noise": ("noise", 0), "guider": ("guider", 0),
                      "sampler": ("sampler", 0), "sigmas": ("sigmas", 0),
                      "latent_image": ("latent", 0)}},
            # Crop Guides takes a plain video latent, so the packed pair splits
            # first; the audio half needs no cropping.
            {"key": "sep", "type": "LTXVSeparateAVLatent", "pos": (4, 0),
             "wire": {"av_latent": ("sc", 0)}},
            {"key": "crop", "type": "LTXVCropGuides", "pos": (4, 1),
             "wire": {"positive": ("guide", 0), "negative": ("guide", 1),
                      "latent": ("sep", 0)}},
            {"key": "vdec", "type": "VAEDecodeTiled", "pos": (5, 1),
             "widgets": {"tile_size": 512, "overlap": 64, "temporal_size": 64,
                         "temporal_overlap": 16},
             "wire": {"samples": ("crop", 2), "vae": ("vvae", 0)}},
            {"key": "adec", "type": "LTXVAudioVAEDecode", "pos": (5, 2),
             "wire": {"samples": ("sep", 1), "audio_vae": ("avae", 0)}},
            {"key": "simg", "type": "SaveImage", "pos": (6, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ltx25_i2v_guide"},
             "wire": {"images": ("vdec", 0)}},
            {"key": "saud", "type": "SaveAudioAdvanced", "pos": (6, 2),
             "widgets": {"filename_prefix": "audio/dgx_monarch_ltx25_i2v_guide",
                         "format": "flac"},
             "wire": {"audio": ("adec", 0)}},
        ],
    },
    "dgx-monarch-ltx25-flf2v": {
        "title": "LTX 2.5 first and last frame to video with sound",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LTX 2.5 first and last frame to video with sound at 1280x704, 121 "
             "frames. Two LTXV Add Guide nodes chain on the video latent: the first "
             "at frame_idx 0 for your opening frame, the second at frame_idx -1, "
             "which counts back from the end. Both append a latent frame, so the "
             "video stream carries 15840 rows here, still even at world 2, and Crop "
             "Guides strips both before the decode. Guide strength is 1.0 on both "
             "guides here, which runs on every topology. Any other value, or a "
             "connected attention_mask, asks for an additive self-attention bias, "
             "and Ulysses re-expresses that bias exactly: the shipped topology auto "
             "resolves to uly2 for LTX at every canvas, so an attenuated guide "
             "renders, while ring and hybrid refuse it before any pixels because the "
             "ring kernel takes no bias argument (docs/TROUBLESHOOTING.md #80). "
             "Topology 'single' and the cfg topologies take it too. The released "
             "recipe runs strength 0.7 on both guides, so a graph diffed against the "
             "stock template will not match at 1.0. The decode reads the sampler's "
             "denoised output, slot 1, as the released first-and-last-frame recipe "
             "does; slot 0 is the plain sample. The sampler is euler, not the "
             "euler_ancestral its three siblings ship, because the released recipes "
             "differ: the official first-and-last-frame graph reaches this sigma "
             "list through Sampler Euler Ancestral at eta 0.0, which adds no "
             "ancestral noise and is plain euler, while the official text-to-video "
             "and image-to-video graphs ask KSampler Select for euler_ancestral and "
             "get comfy's eta 1.0. Point both Load Image nodes at your own frames. "
             + FRAMES + " " +
             GATE + " " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot"
                         ".safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name":
                         "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                         "type": "ltxv"}},
            # The official guidance for a first-and-last-frame prompt: describe the
            # transition, the camera move and the audio, and keep both frames named.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "Use the provided start image as the first frame and "
                                 "the provided end image as the final frame. A single "
                                 "continuous take carries the scene from the first "
                                 "framing to the last, the camera moving steadily and "
                                 "without a cut while the light keeps the same "
                                 "direction and warmth throughout. The motion resolves "
                                 "exactly on the closing composition and settles "
                                 "there. A quiet room tone runs the length of the "
                                 "shot, with a soft rustle as the movement finishes."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "pc game, console game, video game, cartoon, "
                                 "childish, ugly"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "ltxc", "type": "LTXVConditioning", "pos": (2, 2),
             "widgets": {"frame_rate": 24.0},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0)}},
            {"key": "vvae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
            {"key": "avae", "type": "VAELoader", "pos": (0, 5),
             "widgets": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
            {"key": "first", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "example.png", "upload": "image"}},
            {"key": "last", "type": "LoadImage", "pos": (0, 6),
             "widgets": {"image": "example.png", "upload": "image"}},
            {"key": "vlatent", "type": "EmptyLTXVLatentVideo", "pos": (1, 4),
             "widgets": {"width": 1280, "height": 704, "length": 121, "batch_size": 1}},
            {"key": "g1", "type": "LTXVAddGuide", "pos": (2, 3),
             "widgets": {"frame_idx": 0, "strength": 1.0},
             "wire": {"positive": ("ltxc", 0), "negative": ("ltxc", 1),
                      "vae": ("vvae", 0), "latent": ("vlatent", 0),
                      "image": ("first", 0)}},
            # The second guide reads the first one's conditioning, so the two
            # keyframe records stack in the order the model expects.
            {"key": "g2", "type": "LTXVAddGuide", "pos": (3, 3),
             "widgets": {"frame_idx": -1, "strength": 1.0},
             "wire": {"positive": ("g1", 0), "negative": ("g1", 1),
                      "vae": ("vvae", 0), "latent": ("g1", 2),
                      "image": ("last", 0)}},
            {"key": "alatent", "type": "LTXVEmptyLatentAudio", "pos": (1, 5),
             "widgets": {"frames_number": 121, "frame_rate": 24.0, "batch_size": 1},
             "wire": {"audio_vae": ("avae", 0)}},
            {"key": "latent", "type": "LTXVConcatAVLatent", "pos": (3, 4),
             "wire": {"video_latent": ("g2", 2),
                      "audio_latent": ("alatent", 0)}},
            {"key": "sigmas", "type": "ManualSigmas", "pos": (2, 5),
             "widgets": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, "
                                   "0.725, 0.421875, 0.0"}},
            # The released first-and-last-frame recipe reaches this schedule through
            # Sampler Euler Ancestral at eta 0.0, which injects no ancestral noise
            # and reduces to plain euler; the three other 2.5 templates follow their
            # own official graphs, which ask KSampler Select for euler_ancestral and
            # so keep comfy's eta 1.0 default. The recipes differ, not the port.
            {"key": "sampler", "type": "KSamplerSelect", "pos": (1, 6),
             "widgets": {"sampler_name": "euler"}},
            {"key": "noise", "type": "RandomNoise", "pos": (1, 7),
             "widgets": {"noise_seed": 42}},
            {"key": "guider", "type": "DGXMonarchCFGGuider", "pos": (4, 0),
             "widgets": {"cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("g2", 0),
                      "negative": ("g2", 1)}},
            {"key": "sc", "type": "DGXMonarchSamplerCustom", "pos": (4, 1),
             "wire": {"noise": ("noise", 0), "guider": ("guider", 0),
                      "sampler": ("sampler", 0), "sigmas": ("sigmas", 0),
                      "latent_image": ("latent", 0)}},
            {"key": "sep", "type": "LTXVSeparateAVLatent", "pos": (5, 0),
             "wire": {"av_latent": ("sc", 1)}},
            {"key": "crop", "type": "LTXVCropGuides", "pos": (5, 1),
             "wire": {"positive": ("g2", 0), "negative": ("g2", 1),
                      "latent": ("sep", 0)}},
            {"key": "vdec", "type": "VAEDecodeTiled", "pos": (6, 1),
             "widgets": {"tile_size": 512, "overlap": 64, "temporal_size": 64,
                         "temporal_overlap": 16},
             "wire": {"samples": ("crop", 2), "vae": ("vvae", 0)}},
            {"key": "adec", "type": "LTXVAudioVAEDecode", "pos": (6, 2),
             "wire": {"samples": ("sep", 1), "audio_vae": ("avae", 0)}},
            {"key": "simg", "type": "SaveImage", "pos": (7, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ltx25_flf2v"},
             "wire": {"images": ("vdec", 0)}},
            {"key": "saud", "type": "SaveAudioAdvanced", "pos": (7, 2),
             "widgets": {"filename_prefix": "audio/dgx_monarch_ltx25_flf2v",
                         "format": "flac"},
             "wire": {"audio": ("adec", 0)}},
        ],
    },
    "dgx-monarch-hunyuan-image": {
        "title": "HunyuanImage 2.1 text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "HunyuanImage 2.1 text to image at 2048x2048, 20 steps at cfg 3.5. A "
             "17B DiT: auto picks uly2 on a configured pair at every resolution, and "
             "on 2026-10-06 this graph's uly2 render matched the single-GPU output exactly. "
             "The negative comes from Conditioning Zero Out instead of an empty text "
             "encode, which keeps the two conditionings the same length and leaves "
             "cfg-parallel available; with quoted text in play the glyph stream "
             "needs equal-length prompts. With no quoted span in either prompt, a "
             "real unequal negative also batches through the text-mask equalizer "
             "(checked on hardware 2026-08-21). Put any words you want rendered in "
             "double quotes: the tokenizer builds glyph tokens only from quoted "
             "spans, so without them the second text encoder this graph loads does "
             "nothing. The refiner is a separate checkpoint with its own VAE and "
             "latent layout, so it cannot resample this graph's latent: "
             "decode with the 2.1 VAE, re-encode the pixels with "
             "hunyuan_image_refiner_vae_fp16.safetensors, pass both conditionings "
             "through Hunyuan Latent Refiner, then sample the refiner checkpoint "
             "for 4 steps at cfg 1.0. " + GATE + " Expects "
             "hunyuanimage2.1_fp8_e4m3fn.safetensors in models/diffusion_models, "
             "qwen_2.5_vl_7b_fp8_scaled.safetensors plus "
             "byt5_small_glyphxl_fp16.safetensors in models/text_encoders, and "
             "hunyuan_image_2.1_vae_fp16.safetensors in models/vae; the refiner "
             "stage adds hunyuanimage2.1_refiner_fp8_e4m3fn.safetensors and "
             "hunyuan_image_refiner_vae_fp16.safetensors to the same two "
             "folders. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "hunyuanimage2.1_fp8_e4m3fn.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "clip_name2": "byt5_small_glyphxl_fp16.safetensors",
                         "type": "hunyuan_image"}},
            # A long prompt carrying one quoted span: the tokenizer builds byt5
            # glyph tokens only from double-quoted text, so the second encoder this
            # graph loads is inert on a prompt without one.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A weathered brass compass resting on a folded "
                                 "nautical chart in a lighthouse keeper's study, "
                                 "late afternoon light raking across the desk "
                                 "through a salt streaked window. The chart shows "
                                 "hand inked coastlines, depth soundings and a "
                                 "pencilled course line; a small card tucked under "
                                 "the compass rim reads \"TRUE NORTH\" in crisp "
                                 "letterpress capitals. Beside it sit a stub of red "
                                 "wax, a coil of tarred twine and a chipped enamel "
                                 "mug. Warm amber light against cool teal shadow, "
                                 "fine grain in the aged paper, tarnish and "
                                 "fingerprints on the brass, shallow depth of field "
                                 "with the far edge of the desk softening into haze. "
                                 "Photographic realism, 2K detail, natural colour."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 2),
             "wire": {"conditioning": ("pos", 0)}},
            {"key": "latent", "type": "EmptyHunyuanImageLatent", "pos": (2, 3),
             "widgets": {"width": 2048, "height": 2048, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 3.5},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "hunyuan_image_2.1_vae_fp16.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_hunyuan_image"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-hunyuan-video": {
        "title": "HunyuanVideo 1.5 text to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "HunyuanVideo 1.5 text to video at 1280x720, 121 frames (about 5 s at "
             "24 fps), 20 steps at cfg 6.0. An 8.3B DiT with a /16 spatial latent, "
             "so use the Empty HunyuanVideo 1.5 Latent node, not the 1.0 node beside "
             "it in the menu. auto picks uly2 on a configured pair. The "
             "super-resolution checkpoints are a separate stage with an explicit "
             "topology, not part of this graph. Quoted text in the prompt adds ByT5 "
             "glyph tokens, which uly2 handles, but an explicit cfg2 preset then "
             "needs equal-length positive and negative prompts. The negative is "
             "empty, as the official template ships it. Under uly2 the conditional "
             "and unconditional text streams shard separately, and the pad row an "
             "odd stream takes is dropped from attention: on 2026-10-06 a uly2 "
             "one-step render of this file at this shape, with 77 text rows, was "
             "equal to the single-GPU output. " +
             FRAMES + " " + GATE + " Expects "
             "hunyuanvideo1.5_720p_t2v_fp16.safetensors in models/diffusion_models, "
             "qwen_2.5_vl_7b_fp8_scaled.safetensors plus "
             "byt5_small_glyphxl_fp16.safetensors in models/text_encoders, and "
             "hunyuanvideo15_vae_fp16.safetensors in models/vae. Use the 1.5 VAE, not "
             "the 1.0 one. Filenames, shape and sampler values follow the official "
             "ComfyUI 720p text to video template; for the Hunyuan team's quality "
             "setting, their own table asks for shift 9 at 50 steps. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "hunyuanvideo1.5_720p_t2v_fp16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "clip_name2": "byt5_small_glyphxl_fp16.safetensors",
                         "type": "hunyuan_video_15"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A paper airplane released from the top of a "
                                 "skyscraper, gliding through urban canyons, "
                                 "crossing traffic, flying over streets, spiraling "
                                 "upward between buildings. The camera follows the "
                                 "paper airplane's perspective, shooting cityscape "
                                 "in first-person POV, finally flying toward the "
                                 "sunset, disappearing in golden light. Creative "
                                 "camera movement, free perspective, dreamlike "
                                 "colors."},
             "wire": {"clip": ("clip", 0)}},
            # The official template and the vendor CLI both ship an empty negative,
            # and the tokenizer prevents an empty text encode from collapsing.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": ""},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyHunyuanVideo15Latent", "pos": (2, 3),
             "widgets": {"width": 1280, "height": 720, "length": 121,
                         "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 6.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "hunyuanvideo15_vae_fp16.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_hunyuan_video"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-kandinsky5-image": {
        "title": "Kandinsky 5 text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Kandinsky 5 text to image at 1024x1024, 50 steps at cfg 3.5. Model "
             "Sampling SD3 (DGX Monarch) carries shift 3 because the official Image "
             "workflow does, but this checkpoint already defaults to 3, so the node "
             "restates the value instead of changing it: it puts the number where "
             "you can see and change it. In the two Video templates the node does "
             "override the default. The text side is two encoders in one Dual CLIP "
             "Loader, and the Kandinsky text encode node takes the prompt twice, "
             "once per encoder. The latent is the native 16-channel image layout, "
             "which Empty SD3 Latent Image builds. auto picks uly2 at every "
             "resolution. " + GATE + " Expects "
             "kandinsky5lite_t2i.safetensors in models/diffusion_models, "
             "clip_l.safetensors plus qwen_2.5_vl_7b_fp8_scaled.safetensors in "
             "models/text_encoders, and the Flux VAE, ae.safetensors, in models/vae: "
             "the image model uses the Flux latent layout and has no VAE of its "
             "own. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "kandinsky5lite_t2i.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 3.0}, "wire": {"model": ("unet", 0)}},
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "type": "kandinsky5_image"}},
            # Both encoder fields take the official demo prompt.
            {"key": "pos", "type": "CLIPTextEncodeKandinsky5", "pos": (1, 2),
             "widgets": {"clip_l": _KANDINSKY5_IMAGE_PROMPT,
                         "qwen25_7b": _KANDINSKY5_IMAGE_PROMPT},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncodeKandinsky5", "pos": (1, 3),
             "widgets": {"clip_l": "", "qwen25_7b": ""},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 50, "cfg": 3.5},
             "wire": {"model": ("shift", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_kandinsky5_image"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-kandinsky5-video-lite": {
        "title": "Kandinsky 5 Video Lite text to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Kandinsky 5 Video Lite text to video at 768x512, 121 frames (about 5 s "
             "at 24 fps, 31 latent frames), 50 steps at cfg 5.0 with euler_ancestral "
             "on the beta schedule. Video runs SD3 sampling shift 5, which is not "
             "the checkpoint default, so Model Sampling SD3 (DGX Monarch) stays in "
             "the chain; without it the graph does not reproduce the official "
             "workflow. Lite fits resident on one Spark, so FSDP was never exercised "
             "for it; auto picks uly2 on a configured pair. " + FRAMES + " " + GATE +
             " Expects kandinsky5lite_t2v_sft_5s.safetensors in "
             "models/diffusion_models, clip_l.safetensors plus "
             "qwen_2.5_vl_7b_fp8_scaled.safetensors in models/text_encoders, and the "
             "HunyuanVideo VAE, hunyuan_video_vae_bf16.safetensors, in models/vae: "
             "the video models use the HunyuanVideo latent layout and have no VAE of "
             "their own. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "kandinsky5lite_t2v_sft_5s.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("unet", 0)}},
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "type": "kandinsky5"}},
            {"key": "pos", "type": "CLIPTextEncodeKandinsky5", "pos": (1, 2),
             "widgets": {"clip_l": _KANDINSKY5_LITE_PROMPT,
                         "qwen25_7b": _KANDINSKY5_LITE_PROMPT},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncodeKandinsky5", "pos": (1, 3),
             "widgets": {"clip_l": "", "qwen25_7b": ""},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyHunyuanLatentVideo", "pos": (2, 3),
             "widgets": {"width": 768, "height": 512, "length": 121,
                         "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 50, "cfg": 5.0,
                         "sampler_name": "euler_ancestral", "scheduler": "beta"},
             "wire": {"model": ("shift", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "hunyuan_video_vae_bf16.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_kandinsky5_video_lite"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-kandinsky5-video-pro": {
        "title": "Kandinsky 5 Video Pro text to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Kandinsky 5 Video Pro text to video at 768x512, 121 frames (about 5 s "
             "at 24 fps, 31 latent frames), 50 steps at cfg 5.0 with euler_ancestral "
             "on the beta schedule. No official ComfyUI workflow ships for Pro, so "
             "those values come from the official Lite workflow and the vendor's "
             "own Pro config. It is the Lite graph with the Pro checkpoint and the "
             "same SD3 sampling shift 5, which the checkpoint default does not "
             "supply. Pro is the 4096-dim model: if it does not fit resident, set "
             "the Init topology to uly2+fsdp, which shards the weights across the "
             "pair for capacity and takes roughly twice the time. FSDP adds "
             "capacity, never speed, and LoRA on FSDP is unmeasured for Pro. FSDP "
             "also needs a checkpoint stored in bf16: the launch contract reads the "
             "safetensors header on the driver and refuses fp8, int8, fp16 and "
             "mixed-precision files, so that route needs the bf16 Pro file, not a "
             "quantized repack. " +
             FRAMES + " " + GATE + " Expects "
             "kandinsky5pro_t2v_sft_5s.safetensors in models/diffusion_models, "
             "clip_l.safetensors plus qwen_2.5_vl_7b_fp8_scaled.safetensors in "
             "models/text_encoders, and the HunyuanVideo VAE, "
             "hunyuan_video_vae_bf16.safetensors, in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "kandinsky5pro_t2v_sft_5s.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "shift", "type": "DGXMonarchModelSamplingSD3", "pos": (2, 0),
             "widgets": {"shift": 5.0}, "wire": {"model": ("unet", 0)}},
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "type": "kandinsky5"}},
            {"key": "pos", "type": "CLIPTextEncodeKandinsky5", "pos": (1, 2),
             "widgets": {"clip_l": _KANDINSKY5_PRO_PROMPT,
                         "qwen25_7b": _KANDINSKY5_PRO_PROMPT},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncodeKandinsky5", "pos": (1, 3),
             "widgets": {"clip_l": "", "qwen25_7b": ""},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyHunyuanLatentVideo", "pos": (2, 3),
             "widgets": {"width": 768, "height": 512, "length": 121,
                         "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 50, "cfg": 5.0,
                         "sampler_name": "euler_ancestral", "scheduler": "beta"},
             "wire": {"model": ("shift", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "hunyuan_video_vae_bf16.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_kandinsky5_video_pro"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-cogvideox-t2v": {
        "title": "CogVideoX text to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "CogVideoX text to video at 720x480, 45 frames (about 5.6 s at 8 fps), "
             "50 steps at cfg 6.0. This family pads every conditioning to a fixed "
             "226 tokens, so the positive and the negative always match in length "
             "and cfg2 runs. The auto table has no cfg2 row for it, though, so auto "
             "picks uly2 at every resolution; for the CFG split, set the Init "
             "topology to cfg2 by hand. The 1.5 card specifies 1360x768 for text to "
             "video, and its image to video rule is short side 768, long side "
             "between 768 and 1360 and divisible by 16. The latent frame count must "
             "be even, because 1.5 patches two latent frames at once and the "
             "transformer pads an odd count by wrapping frame 0 around: pick 45, "
             "48, 53, 56, 77, 80 or 88 instead of the vendor's 8N+1 lengths, which "
             "all give an odd count. Change the three latent widgets together. A "
             "CogVideoX 1.0-layout LoRA skips every key on a 1.5 checkpoint and "
             "raises no error. " + FRAMES + " " + GATE +
             " Expects CogVideoX_1_5_5b_T2V_bf16.safetensors in "
             "models/diffusion_models, t5xxl_fp16.safetensors in "
             "models/text_encoders, and cogvideox_vae_bf16.safetensors in models/vae. " +
             DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "CogVideoX_1_5_5b_T2V_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "t5xxl_fp16.safetensors", "type": "cogvideox"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "a hot air balloon rising over a patchwork of "
                                 "fields at dawn, slow ascent"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low quality, distorted, watermark, text"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyHunyuanLatentVideo", "pos": (2, 3),
             "widgets": {"width": 720, "height": 480, "length": 45,
                         "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 50, "cfg": 6.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "cogvideox_vae_bf16.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_cogvideox"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-cogvideox-i2v": {
        "title": "CogVideoX image to video",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "CogVideoX image to video at 1360x768, 45 frames (about 5.6 s at 8 fps), "
             "50 steps at cfg 6.0. Point Load Image at your own first frame; the "
             "shipped filename is a placeholder, and Image Scale puts it on the "
             "latent canvas before the VAE sees it. Comfy ships no CogVideoX image "
             "node, so the bridge is the stock Instruct Pix To Pix Conditioning "
             "node: it encodes that one frame into both conditionings as the concat "
             "latent, and the model zero pads it to the length of the clip, as the "
             "vendor pipeline does. Its LATENT output stays unwired because it is "
             "one frame of zeros; Empty Hunyuan Latent Video builds the real canvas. "
             "An unwired image turns the render into text to video without an "
             "error. The 1.5 image to video card asks for the short side at 768 and "
             "the long side between 768 and 1360 and divisible by 16, and the latent "
             "frame count must be even because 1.5 patches two latent frames at "
             "once, so 45 frames is safe where the vendor's 81 is not; 77 and 80 are "
             "the longer safe lengths. This family pads every conditioning to a "
             "fixed 226 tokens, so cfg2 runs, but the auto table has no cfg2 row for "
             "it and picks uly2 at every resolution; for the CFG split, set the Init "
             "topology to cfg2 by hand. A CogVideoX 1.0-layout LoRA skips every key "
             "on a 1.5 checkpoint and raises no error, and the 1.0 files sit beside "
             "the 1.5 one under nearly the same name. The ComfyUI-layout BF16 I2V "
             "auto/Ulysses path is HW-scoped: cold and warm 45-frame renders "
             "completed, and the decoded one-step probe read NRMS 0.015932 against "
             "its local reference. Diffusers-layout loading and inpaint are not "
             "covered. " + FRAMES +
             " " + GATE + " Expects CogVideoX_1_5_5b_I2V_bf16.safetensors in "
             "models/diffusion_models, t5xxl_fp16.safetensors in "
             "models/text_encoders, and cogvideox_vae_bf16.safetensors in models/vae. " +
             DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "CogVideoX_1_5_5b_I2V_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "t5xxl_fp16.safetensors", "type": "cogvideox"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A weathered wooden sailboat drifts across a calm "
                                 "turquoise bay in the late afternoon. The single sail "
                                 "fills slowly as a light breeze crosses the water and "
                                 "the hull rocks with an even, unhurried rhythm. "
                                 "Sunlight scatters into bright moving ripples along "
                                 "the surface, and a thin trail of white foam widens "
                                 "behind the stern. The camera holds a steady wide "
                                 "shot and drifts almost imperceptibly to the right, "
                                 "keeping the boat just off center. Distant green "
                                 "headlands soften into a warm haze, seabirds glide in "
                                 "slow arcs overhead, and the whole scene keeps a "
                                 "quiet, cinematic calm."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low quality, distorted, watermark, text"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "cogvideox_vae_bf16.safetensors"}},
            {"key": "image", "type": "LoadImage", "pos": (0, 5),
             "widgets": {"image": "example.png", "upload": "image"}},
            # Instruct Pix To Pix Conditioning crops to a multiple of 8 but never
            # resizes, so the pixels are put on the canvas here instead of letting
            # concat_cond resize the encoded latent.
            {"key": "scale", "type": "ImageScale", "pos": (1, 5),
             "widgets": {"upscale_method": "lanczos", "width": 1360, "height": 768,
                         "crop": "center"},
             "wire": {"image": ("image", 0)}},
            # This is the only stock node that sets concat_latent_image from a
            # plain image with no mask. The CogVideoX VAE is 3D, so one image
            # encodes to a one-frame latent and CogVideoX.concat_cond zero pads it
            # to the clip.
            # Slot 2 (its own latent) is one frame of zeros and stays unwired.
            {"key": "i2v", "type": "InstructPixToPixConditioning", "pos": (2, 2),
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0),
                      "vae": ("vae", 0), "pixels": ("scale", 0)}},
            {"key": "latent", "type": "EmptyHunyuanLatentVideo", "pos": (2, 4),
             "widgets": {"width": 1360, "height": 768, "length": 45,
                         "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 50, "cfg": 6.0},
             "wire": {"model": ("unet", 0), "positive": ("i2v", 0),
                      "negative": ("i2v", 1), "latent_image": ("latent", 0)}},
            # A full-frame decode at this canvas prices well past a pair's unified
            # memory, so the tiled decoder is the shipped path.
            {"key": "dec", "type": "VAEDecodeTiled", "pos": (4, 1),
             "widgets": {"tile_size": 512, "overlap": 64, "temporal_size": 64,
                         "temporal_overlap": 16},
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_cogvideox_i2v"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-flux2-t2i": {
        "title": "Flux 2 text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "uly2", "mode": "auto",
                         "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Flux 2 text to image at 1024x1024, 20 steps at cfg 1.0 with the "
             "guidance embed at 4.0. Flux 2 is guidance distilled, so cfg stays at "
             "1.0, and the Conditioning Zero Out negative gives the sampler a real "
             "unconditional; it steers nothing. The Flux Guidance node sets prompt "
             "adherence instead, so change guidance, not cfg. The latent is the Flux "
             "2 layout (128 channels, /16 spatial), which only Empty Flux 2 Latent "
             "builds. The topology is resident uly2 on the fp8mixed repack, "
             "measured 2026-08-21 at 1-step NRMS 0.016 (1024) and 0.022 (1536) "
             "against DP2 with a 20-step render; on 2026-10-06 it was "
             "equal to the dp2 output at 1024 and 1040. The bf16 release with the "
             "`uly2+fsdp` preset carries the 2026-07-28 evidence instead. This graph "
             "ships the resident path, and the FSDP launch contract refuses every "
             "quantized file, so switching to FSDP means switching to the bf16 "
             "checkpoint too. " + GATE +
             " Expects flux2_dev_fp8mixed.safetensors in "
             "models/diffusion_models, mistral_3_small_flux2_bf16.safetensors in "
             "models/text_encoders, and flux2-vae.safetensors in models/vae. " +
             DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             # Resident uly2 wants the fp8mixed repack (2026-08-21 evidence);
             # the literal bf16 release belongs to the `uly2+fsdp` preset and
             # its 2026-07-28 scope (benchmark/flux2_cross_topology_matrix.toml).
             "widgets": {"unet_name": "flux2_dev_fp8mixed.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "mistral_3_small_flux2_bf16.safetensors",
                         "type": "flux2"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "high fashion, vintage couture, street photography, "
                                 "luxury fashion shoot, neo brutalist architecture, "
                                 "pastel paints"},
             "wire": {"clip": ("clip", 0)}},
            # The guidance embed, not cfg, is what steers a distilled Flux 2. With
            # no such node comfy falls back to 3.5; the official graph sets 4.0.
            # Zero Out reads the guided conditioning, so both branches carry the
            # same guidance key and the same token count.
            {"key": "guid", "type": "FluxGuidance", "pos": (2, 1),
             "widgets": {"guidance": 4.0},
             "wire": {"conditioning": ("pos", 0)}},
            {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 2),
             "wire": {"conditioning": ("guid", 0)}},
            {"key": "latent", "type": "EmptyFlux2LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("guid", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "flux2-vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_flux2"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-ideogram4-t2i": {
        "title": "Ideogram 4 text to image (dual model CFG)",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "uly2", "mode": "auto",
                         "auto_gate": "first_use", "sync_ulysses": False}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Ideogram 4 text to image at 1024x1024, 20 steps at cfg 7.0. Two things "
             "set this family apart. First, the unconditional pass is a separate "
             "checkpoint, so there is no CFG batch to split: the second loader holds "
             "the unconditional model, the Dual Model Guider drives both, and the "
             "negative input stays unconnected for the text-free unconditional. "
             "Both models are resident at once, so budget memory for the pair. "
             "Second, the topology is uly2, the row auto resolves to for this "
             "family: on 2026-08-11 uly2 matched the DP2 reference at 1-step NRMS "
             "0.000 on this exact fp8 pair, and on 2026-10-06 this graph's uly2 "
             "render matched the single-GPU output exactly on the bf16, fp8 and int8 pairs. "
             "Ring2 measured 0.102 against the 0.100 limit and is an explicit "
             "diagnostic only, and unlisted quantizations inherit the head-divisible "
             "choice with no hardware claim of their own. The prompt is the "
             "structured JSON this family expects, not a plain sentence: plain text "
             "is accepted but performs worse and raises the safety filter's "
             "false-positive rate, and the caption is validated against a schema, "
             "so compositional_deconstruction with its background and elements is "
             "required and the key order inside style_description is fixed. The "
             "official presets drop guidance to 3 for the final polish steps, which "
             "the DGX Monarch guider cannot express, so this template runs the "
             "vendor's constant-guidance mode at cfg 7.0. Sigmas come from the stock "
             "Ideogram 4 Scheduler, so keep its width and height equal to the "
             "latent's. sync_ulysses is off for this family: dropping the "
             "per-all-to-all device sync rendered 24.0 s against 26.0 s with it on, "
             "with 1-step NRMS 0.069 against DP2 for both sync settings "
             "(2026-08-25). A gate verdict covers one sync_ulysses setting, so the "
             "first render with it off runs its own identity gate before any "
             "pixels. " + GATE +
             " Expects ideogram4_fp8_scaled.safetensors plus "
             "ideogram4_unconditional_fp8_scaled.safetensors in "
             "models/diffusion_models, qwen3vl_8b_fp8_scaled.safetensors in "
             "models/text_encoders, and the Flux 2 VAE, flux2-vae.safetensors, in "
             "models/vae: this family uses the Flux 2 latent layout and has no VAE "
             "of its own, and all four files come from the one repack. The VAE and "
             "the decode tail come from the stock node definitions."},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "ideogram4_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "uncond", "type": "DGXMonarchUncondUNETLoader", "pos": (1, 1),
             "widgets": {"unet_name":
                         "ideogram4_unconditional_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
                         "type": "ideogram4"}},
            # One line of compact JSON, valid against the official caption schema:
            # compositional_deconstruction with background and elements is required,
            # style_description carries aesthetics and lighting and exactly one of
            # photo or art_style, and the photo key order is fixed. bbox is
            # [y_min, x_min, y_max, x_max] over 0 to 1000 from the top left.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text":
                         '{"high_level_description":"A close-up photograph of a '
                         'chalkboard menu sign propped on a wooden bakery counter '
                         'beside a tray of fresh croissants.","style_description":'
                         '{"aesthetics":"warm, rustic, inviting","lighting":"soft '
                         'morning window light from the left with gentle falloff",'
                         '"photo":"50mm, f/2.8, shallow depth of field, eye level",'
                         '"medium":"photograph","color_palette":["#2B2018",'
                         '"#E8D5B7","#C98A3C","#F5EFE6","#6B4A2F"]},'
                         '"compositional_deconstruction":{"background":"A warm '
                         'wooden bakery counter with a softly blurred shelf of bread '
                         'loaves and a copper kettle behind it.","elements":'
                         '[{"type":"text","bbox":[120,180,520,820],"text":"FRESH '
                         'BAKED DAILY","desc":"Hand lettered white chalk capitals on '
                         'a black chalkboard, set in three stacked centered lines '
                         'inside a thin chalk border, with a small wheat sprig drawn '
                         'beneath."},{"type":"obj","bbox":[560,120,940,880],"desc":'
                         '"A shallow metal tray of golden croissants with flaky '
                         'layered crusts, dusted lightly with flour, angled toward '
                         'the camera."}]}}'},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyFlux2LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            # mu 0.0 with std 1.75 at 20 steps is the vendor's V4_DEFAULT_20 preset.
            # Its fast preset is 12 steps at mu 0.5, and no official preset runs
            # fewer steps.
            {"key": "sched", "type": "Ideogram4Scheduler", "pos": (2, 4),
             "widgets": {"steps": 20, "width": 1024, "height": 1024, "mu": 0.0,
                         "std": 1.75}},
            {"key": "sampler", "type": "KSamplerSelect", "pos": (1, 4),
             "widgets": {"sampler_name": "euler"}},
            {"key": "noise", "type": "RandomNoise", "pos": (1, 5),
             "widgets": {"noise_seed": 42}},
            {"key": "guider", "type": "DGXMonarchDualModelGuider", "pos": (3, 0),
             "widgets": {"cfg": 7.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "model_negative": ("uncond", 0)}},
            {"key": "sc", "type": "DGXMonarchSamplerCustom", "pos": (3, 1),
             "wire": {"noise": ("noise", 0), "guider": ("guider", 0),
                      "sampler": ("sampler", 0), "sigmas": ("sched", 0),
                      "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "flux2-vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("sc", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ideogram4"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-flux1-t2i": {
        "title": "Flux 1 text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto",
                         "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Flux 1 Dev text to image at 1024x1024, 20 steps at cfg 1.0, guidance "
             "embed 3.5. Flux 1 Dev is guidance distilled, so cfg stays at 1.0, and "
             "the Conditioning Zero Out negative gives the sampler an unconditional; "
             "it steers nothing. The Flux Guidance node sets prompt adherence, and "
             "its 3.5 is also what comfy applies when the graph has no such node. "
             "topology=auto runs this on one Spark and splits it across the pair "
             "once a cluster.toml is configured. At 1.05 MP no flux row of the auto "
             "table applies, so auto takes the generic uly2 fallback, the mode the "
             "2026-08-11 Flux 1 Dev hardware run covers; on 2026-10-06 this graph's uly2 "
             "render matched the single-GPU output exactly. Sequence-parallel shards the "
             "text and the image stream separately. 1024x1024 and the shipped prompt "
             "both divide by two, so nothing pads. Since 2026-09-09 a stream that "
             "does not divide is padded, and its pad rows are excluded from every "
             "attention call instead of refused. The fidelity legs for that path "
             "ran on 2026-09-09 and passed: 1-step NRMS 0.006 on the padded image "
             "stream and 0.005 on the padded text stream, each against its own "
             "divisible companion. A prompt past the T5 floor of 256 tokens to an "
             "odd length pads the text stream, and an odd canvas such as 1448 "
             "square, 91 by 91 = 8281 rows, pads the image stream. For Flux 1 "
             "Schnell, point the loader at flux1-schnell.safetensors and set steps "
             "to 4, nothing else; the guidance value is then ignored, because that "
             "checkpoint carries no guidance embed. Schnell has its own record: on "
             "2026-08-21 uly2 matched DP2 at 1-step NRMS 0.048 at 1024x1024 and its "
             "4-step render passed; no padded Schnell render has been measured. "
             "XLabs-format LoRAs do not load here; use a converted release. " + GATE +
             " Expects flux1-dev.safetensors in models/diffusion_models, "
             "clip_l.safetensors and t5xxl_fp16.safetensors in models/text_encoders, and "
             "ae.safetensors in models/vae; flux1-dev-fp8.safetensors is an all in one "
             "checkpoint for models/checkpoints, not a diffusion-model file. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "flux1-dev.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "DualCLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A fairy tale scene of a young girl with silver "
                                 "curly hair wearing a delicate white dress, standing "
                                 "among crystal butterflies and glowing glass roses. "
                                 "The scene is filled with soft magical light, like a "
                                 "dream from a fantasy world."},
             "wire": {"clip": ("clip", 0)}},
            # The guidance embed, not cfg, is what steers a distilled Flux 1. 3.5 is
            # both the node default and the value comfy applies with no node in the
            # graph, so this node changes nothing until the value is moved. Set it
            # explicitly: the stock default map in _default_for carries no
            # `guidance` key.
            {"key": "guid", "type": "FluxGuidance", "pos": (2, 2),
             "widgets": {"guidance": 3.5},
             "wire": {"conditioning": ("pos", 0)}},
            # Zero out the guided conditioning, so both branches carry the same
            # guidance key and the same token count. At cfg 1.0 comfy skips the
            # unconditional pass; the sampler still requires the input.
            {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 3),
             "wire": {"conditioning": ("guid", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (1, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("guid", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_flux1"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-chroma-t2i": {
        "title": "Chroma 1 HD text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Chroma 1 HD text to image at 1024x1024, 26 steps at cfg 3.5 with euler "
             "and the beta scheduler. topology=auto runs it on one Spark. On a "
             "configured pair below 1.2 MP with cfg above 1.0 it picks cfg2, a grant "
             "that covers bf16, fp8 and int8 checkpoints; "
             "1024x1024 is 1.05 MP, so the shipped fp8 graph lands on cfg2. The "
             "negative is a real prompt: cfg-parallel pads the shorter side and "
             "masks the appended rows, aligning the masked key extent to a multiple "
             "of 8, so unequal prompt lengths batch cleanly on "
             "cfg2. At 1.2 MP and above auto selects uly2, requalified on hardware "
             "2026-08-21 for streams that divide the sequence-parallel degree "
             "exactly. A Chroma stream that does not divide it is "
             "padded and its pad rows are dropped from every attention call, so an "
             "odd prompt and a blank negative both reach uly2 instead of refusing. "
             "The two counts do not have to match; cfg2 is the path that pads and "
             "masks a mismatch. The shipped pair counts 100 and 28, both even on "
             "purpose: that exclusion has a full-sequence point only under "
             "Ulysses, so on ring or hybrid a padded stream stops with the "
             "waivable ring_pad card. The sharded kernel still refuses an "
             "effective attention mask, so send a masked or regional "
             "conditioning to cfg2 or to a one-GPU run. " + GATE + " Expects "
             "Chroma1-HD-fp8mixed.safetensors in models/diffusion_models, "
             "t5xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders, and the "
             "Flux VAE, ae.safetensors, in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "Chroma1-HD-fp8mixed.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "chroma"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "This is a nature documentary close-up photograph of "
                                 "the right side of the face of a tiger. The photograph "
                                 "is centered on its highly detailed and speckled eye "
                                 "surrounded by intricately detailed fur. Overlaid at "
                                 "the center of the image is a title text that says "
                                 "\"CHROMA1-HD\" in large white 3D letters. Amateur "
                                 "photography. Unfiltered. Real life. Natural light. "
                                 "Subtle shadows. Sharp focus. "},
             "wire": {"clip": ("clip", 0)}},
            # A real text negative: since the masked-extent fix (2026-08-21),
            # unequal prompt lengths batch on cfg2. Checked on hardware through
            # this template the same day: an unequal pair, padded and aligned to
            # a multiple of 8, rendered with zero faults. This one counts 28
            # tokens against the positive's 100, so the pair stays unequal and
            # both halves stay even. uly2 does not need even counts: since
            # 2026-09-03 chroma pads a text stream that does not divide and drops
            # the pad rows from every attention call. That exclusion works only
            # under Ulysses; on ring or hybrid a padded stream takes the waivable
            # ring_pad card, so both prompts stay even (made so on 2026-09-02).
            # Count them with tools/count_t5_tokens.py before changing either one.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (2, 2),
             "widgets": {"text": "low quality, blurry, watermark, text artifacts, "
                                 "oversaturated colors, deformed anatomy, noisy "
                                 "background"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            # scheduler is spelled out: the generator's comfy stub defaults the
            # combo to its first entry, and beta is the sigma schedule the
            # official Chroma template's BasicScheduler names.
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 26, "cfg": 3.5, "sampler_name": "euler",
                         "scheduler": "beta"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_chroma"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-radiance-t2i": {
        "title": "Chroma 1 Radiance text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Chroma 1 Radiance text to image at 1024x1024, 30 steps at cfg 3.5 with "
             "euler and the beta schedule; the official graph runs beta at alpha and "
             "beta 0.4 while the KSampler's beta is fixed at 0.6, so the curve is "
             "close but not identical. The checkpoint reports family chroma at "
             "runtime, but this is the Radiance row, validated on hardware "
             "2026-08-21 (cfg2 1-step NRMS 0.043 plus a 30-step render). The "
             "Conditioning Zero Out negative is deliberate: Radiance denoises in "
             "pixel space, where the pad-mask path fails in stock mask upscaling, so "
             "an unequal prompt pair typed-refuses before dispatch; the zeroed "
             "negative keeps the token counts equal. Megapixels count the pixel "
             "grid here, so 1024x1024 prices as about 1.0 MP and topology=auto "
             "picks cfg-parallel on a configured pair; at 1.2 MP and above auto asks "
             "for Ulysses. A stream that does not divide the split is padded there "
             "and its pad rows are dropped from every attention call: Radiance runs "
             "the Chroma adapter, which excludes them. No "
             "Radiance render above 1.2 MP has been measured, so keep the canvas at "
             "1024x1024. " + GATE +
             " Expects chroma-radiance-x0.safetensors in models/diffusion_models and "
             "t5xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders. There is "
             "no VAE file to fetch: pick pixel_space in the VAE Loader list, and the "
             "decode passes the pixels through. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "chroma-radiance-x0.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                         "type": "chroma"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A surreal, high-contrast macro photograph of a "
                                 "small cluster of delicate, slightly withered "
                                 "flowers against a solid black background. Thin, "
                                 "green-brown stems rise from the bottom, presenting "
                                 "blossoms with papery, off-white and brown petals. "
                                 "The most striking feature is the ethereal, "
                                 "rainbow-colored light that emanates from the "
                                 "flowers like gentle flames or an aura. This "
                                 "translucent, iridescent glow is most prominent "
                                 "around the intricate, golden-tipped stamens and "
                                 "contains vibrant hues of purple, blue, green, "
                                 "yellow, and orange, creating a magical and "
                                 "mesmerizing effect."},
             "wire": {"clip": ("clip", 0)}},
            # Conditioning Zero Out, not a written negative: auto picks cfg2 at
            # this resolution, and the pad-plus-mask path for unequal prompt
            # lengths dies in stock mask upscaling on a pixel-space grid
            # (measured 2026-08-21; docs/MODELS.md Radiance row). Zeroing the
            # positive gives an equal-length real unconditional input, so cfg2
            # batches unmasked whatever the prompt says.
            {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 2),
             "wire": {"conditioning": ("pos", 0)}},
            # Pixel space: this node emits a 3-channel image-sized tensor, so the
            # driver prices it 1:1 rather than at the /8 VAE scale. No other Empty
            # Latent builds it.
            {"key": "latent", "type": "EmptyChromaRadianceLatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            # No model sampling node. Chroma declares multiplier 1.0 and shift
            # defaults to 1.0, which is exactly the official Flow Shift setting;
            # DGXMonarchModelSamplingSD3 would apply multiplier 1000 instead.
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 30, "cfg": 3.5, "sampler_name": "euler",
                         "scheduler": "beta"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "pixel_space"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_radiance"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-longcat-t2i": {
        "title": "LongCat-Image text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "LongCat-Image text to image at 1024x1024, 20 steps at cfg 4.0; the "
             "vendor's own card asks for 50 steps, so raise steps first for "
             "quality. This family runs real CFG, so the negative is a real text "
             "encode, not Conditioning Zero Out, and auto splits that CFG batch "
             "below 1.2 MP and moves to uly2 sequence-parallel at and above it. The "
             "text encoder pads every prompt to 512 tokens, positive and negative "
             "alike, so the two conditionings always match in length and cfg2 stays "
             "available whatever you type. To render readable words, put them in "
             "quotation marks: the model encodes a quoted span character by "
             "character, and without the quotes that path never runs. The official "
             "ComfyUI graph also patches the model with CFG Norm to hold saturation "
             "down at cfg 4.0, and the stock patch node cannot take a mesh model, so "
             "this graph leaves it out and relies on the negative prompt, which "
             "lists oversaturated for that reason. " + GATE + " Expects longcat_image_bf16.safetensors in "
             "models/diffusion_models, qwen_2.5_vl_7b_fp8_scaled.safetensors in "
             "models/text_encoders, and the Flux VAE, ae.safetensors, in models/vae: "
             "this family uses the Flux latent layout and has no VAE of its own. "
             + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "longcat_image_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "type": "longcat_image"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "High-fashion portrait of a man with a shaved head and "
                                 "stubble, head tilted back and gazing upward. His skin "
                                 "and thick-knitted dark green turtleneck sweater are "
                                 "bathed in a monochromatic teal-green light, creating a "
                                 "uniform, matte finish. He wears round, thin-framed "
                                 "sunglasses with reflective amber-orange lenses that "
                                 "catch the light. The background is a solid, vibrant "
                                 "burnt orange, creating a bold, high-contrast color "
                                 "palette. The lighting is hard and directional, casting "
                                 "sharp shadows on his face. Hyper-detailed, "
                                 "photorealistic, sharp focus on facial features and "
                                 "fabric texture, editorial photography, 8K."},
             "wire": {"clip": ("clip", 0)}},
            # cfg 4.0 is real guidance, so the unconditional does real steering
            # work: this family takes a written negative, not Conditioning Zero
            # Out. The tokenizer pads both sides to 512 tokens, so the asymmetric
            # lengths cost nothing under cfg2.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low resolution, oversaturated, harsh lighting, "
                                 "messy composition, distorted face, extra fingers, bad "
                                 "anatomy, cheap jewelry, plastic texture, cartoon, "
                                 "illustration, anime, watermark, text, logo"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 4.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_longcat"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-qwen-image-t2i": {
        "title": "Qwen-Image 2512 text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Qwen-Image text to image at 1328x1328, 50 steps at cfg 4.0 with euler "
             "and simple: the base 2512 checkpoint is not guidance distilled, so cfg "
             "4.0 is real CFG and the negative prompt steers, and a 20B model at "
             "this size for 50 steps is a long first render. There is no model "
             "sampling node here: the official ComfyUI example sets shift 3.1 "
             "through a multiplier 1.0 AuraFlow patch this repo has no node for, so "
             "the render runs the checkpoint's own shift of 1.15, the value the "
             "recorded legs ran. On a configured pair topology=auto picks Ulysses at "
             "1.2 MP and above and the generic uly2 fallback below it, so cfg2 runs "
             "only as an explicit preset; docs/MODELS.md records Ulysses and cfg2 "
             "evidence for the base 2512 FP8 file. Qwen-Image-Edit 2511 is out of "
             "scope, and under Ulysses the sharded forward refuses any render that "
             "carries a text attention mask. Auto keeps the Init node's attention "
             "kernel on this family and does not switch it to SAGE: on this graph "
             "SAGE under Ulysses read 1-step NRMS 0.162 against one GPU in the "
             "September 2026 campaign, over the 0.10 limit. An explicit SAGE choice "
             "still runs as selected and needs sageattention on every worker, "
             "because a first configure that cannot import it raises instead of "
             "falling back. This canvas is the official one, and at 1328 square it "
             "patchifies to 83 by 83 image tokens, an odd 6889. Under pure Ulysses "
             "the pad row of that odd stream is dropped from attention: on "
             "2026-10-06 uly2 matched the single-GPU output exactly at 1328 and at 1024. "
             "1024 square gives 64 by 64, an even 4096, and is the canvas the row's "
             "fresh run used. "
             + GATE +
             " Expects qwen_image_2512_fp8_e4m3fn.safetensors in "
             "models/diffusion_models, qwen_2.5_vl_7b_fp8_scaled.safetensors in "
             "models/text_encoders, and qwen_image_vae.safetensors in models/vae. "
             + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "type": "qwen_image"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "Urban alleyway at dusk. Tall, statuesque "
                                 "high-fashion model striding elegantly, mid distant "
                                 "full body shot from an angular perspective, "
                                 "cinematic/editorial with bold contrasts and tactile "
                                 "materials. They wear a rose-gold metallic trench "
                                 "coat with deconstructed elements over a black "
                                 "long-sleeved turtleneck with subtle texture; paired "
                                 "with forest-green pleated pants with raw hems and a "
                                 "soft texture. Long braided dark hair, medium "
                                 "complexion. They carry a vibrant yellow designer "
                                 "handbag with geometric details and a structured "
                                 "silhouette. White architectural sneakers with bold "
                                 "geometric cutouts. Bold, high-contrast, tactile, "
                                 "urban-grit meets high-fashion impact, extreme "
                                 "clarity, extreme layering, post-processing with "
                                 "transparent light-transmitting ultra-smooth "
                                 "high-definition film effect, removing all noise and "
                                 "grain, removing all blur, removing all vintage feel, "
                                 "removing all roughness, drawn with 32K pixel "
                                 "precision, unparalleled fine line drawing of every "
                                 "single detail, the entire image like a brand new "
                                 "photograph, photorealistic"},
             "wire": {"clip": ("clip", 0)}},
            # Qwen-Image is not guidance distilled, so cfg 4.0 is a real CFG and this
            # encode steers. The official negative is Chinese; this is an English
            # rendering of it.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "low resolution, low quality, deformed limbs, "
                                 "deformed fingers, oversaturated, waxy skin, faces "
                                 "with no detail, overly smooth, AI look, messy "
                                 "composition, blurry or distorted text"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (2, 3),
             "widgets": {"width": 1328, "height": 1328, "batch_size": 1}},
            # No model sampling node: the only one this repo has patches with
            # multiplier 1000 (ModelSamplingSD3), and Qwen-Image is a multiplier 1.0
            # family whose timestep projection already scales by 1000 internally.
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 50, "cfg": 4.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "qwen_image_vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_qwen_image"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-qwen-image21-t2i": {
        "title": "Qwen Image 2.1 RGBA text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Qwen Image 2.1 text to image: the pinned ComfyUI 2026-09-28 recipe is "
             "1024x1024, 25 Euler/simple steps, cfg 1.0. This family uses its native "
             "Text Encode Qwen Image 2.1 node, whose positive, negative and empty "
             "64-channel /16 latent outputs stay together. The Qwen Image 2.1 VAE "
             "decodes four channels, so Save Image writes an RGBA PNG. Expects qwen_image_2.1_bf16"
             ".safetensors in models/diffusion_models, qwen3vl_8b_bf16.safetensors in "
             "models/text_encoders, and qwen_image_2.1_vae_bf16.safetensors in models/vae. "
             "This graph leaves the Qwen prefix KV cache off. Do not insert stock "
             "Qwen Image 2.1 Cache here: its MODEL port cannot keep the resident "
             "DGXM_MODEL ownership contract. The optional DGX Qwen Image 2.1 Cache "
             "node turns on a replicated worker cache only when a custom graph asks "
             "for it; its default storage is lossless, and its int8/int4 native K/V "
             "compression is approximate. docs/MODELS.md records its limited native "
             "mode-equivalence scope, the exact 25-step T2I, masked-edit and "
             "ten-reference controls, and the classic three-step default/int8/int4 "
             "cache equivalence. These results do not extend past their named "
             "inputs, resolution, topology or FSDP scope. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "qwen_image_2.1_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image"}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 3),
             "widgets": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            {"key": "cond", "type": "TextEncodeQwenImage21", "pos": (1, 2),
             "widgets": {"prompt": "A quiet, sunlit studio with a ceramicist shaping a "
                                      "blue vase at a wooden wheel; documentary photograph, "
                                      "soft window light, natural materials, fine detail.",
                         "negative_prompt": "blurry, low quality, watermark, text, logo", "resolution": 1024},
             "wire": {"clip": ("clip", 0), "vae": ("vae", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("cond", 0),
                      "negative": ("cond", 1), "latent_image": ("cond", 2)}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_qwen_image21_t2i"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-qwen-image21-edit-rgba": {
        "title": "Qwen Image 2.1 ten-reference RGBA edit",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Qwen Image 2.1 native edit with ten ordered references. Load Image "
             "returns RGB plus its inverse-alpha MASK, so every reference first passes "
             "through Join Image With Alpha. Text Encode Qwen Image 2.1 then composites "
             "that alpha over white for the vision tower while passing all four RGBA "
             "channels to the VAE reference latent. image_1 is the canvas and "
             "sets the output size; image_2 through image_10 are ordered references, "
             "with image_10 named as an optional RGBA mask/reference. Keep this order: "
             "the model records each insertion index. The sampler is the native 25-step "
             "Euler/simple cfg 1.0 recipe and Save Image writes PNG. It uses the same "
             "bf16 DiT, Qwen3-VL-8B encoder and Qwen Image 2.1 VAE as the text graph: "
             "qwen_image_2.1_bf16.safetensors in models/diffusion_models, "
             "qwen3vl_8b_bf16.safetensors in models/text_encoders, and "
             "qwen_image_2.1_vae_bf16.safetensors in models/vae. "
             "This graph leaves prefix caching off; do not add stock Qwen Image 2.1 "
             "Cache to a DGX graph. The optional DGX cache node has lossless default "
             "storage, and its int8/int4 K/V compression modes are approximate. "
             "docs/MODELS.md records its limited native mode-equivalence scope, the "
             "exact 25-step T2I, masked-edit and ten-reference controls, and the "
             "classic three-step default/int8/int4 cache equivalence. These results "
             "do not extend past their named inputs, resolution, topology or FSDP "
             "scope. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "qwen_image_2.1_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image"}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 3),
             "widgets": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            *[{"key": f"ref{i}", "type": "LoadImage", "pos": (1 + (i - 1) % 5, 4 + (i - 1) // 5),
               "widgets": {"image": ("qwen21_canvas_rgba.png" if i == 1 else
                                     "qwen21_mask_rgba.png" if i == 10 else
                                     f"qwen21_reference_{i:02d}_rgba.png")}}
              for i in range(1, 11)],
            *[{"key": f"rgba{i}", "type": "JoinImageWithAlpha", "pos": (1 + (i - 1) % 5, 6 + (i - 1) // 5),
               "wire": {"image": (f"ref{i}", 0), "alpha": (f"ref{i}", 1)}}
              for i in range(1, 11)],
            {"key": "cond", "type": "TextEncodeQwenImage21", "pos": (2, 2),
             "widgets": {"prompt": "Edit image_1 while preserving its subject and layout. "
                                      "Use image_2 through image_9 as ordered visual references. "
                                      "Use image_10's alpha-matted region as the change guide: replace "
                                      "only that region with hand-painted blue-and-gold ceramic detail, "
                                      "keeping edges clean and the unmasked composition unchanged.",
                         "negative_prompt": "blurry, low quality, watermark, text, logo", "resolution": 1024},
             "wire": {"clip": ("clip", 0), "vae": ("vae", 0),
                      **{f"images.image_{i}": (f"rgba{i}", 0) for i in range(1, 11)}}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (4, 2),
             "widgets": {"steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("cond", 0),
                      "negative": ("cond", 1), "latent_image": ("cond", 2)}},
            {"key": "dec", "type": "VAEDecode", "pos": (5, 2),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (6, 2),
             "widgets": {"filename_prefix": "dgx_monarch_qwen_image21_edit_rgba"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-qwen-image21-edit-mask-rgba": {
        "title": "Qwen Image 2.1 masked RGBA reference edit",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Qwen Image 2.1 single-reference masked edit. Load Image supplies the "
             "source RGB; Load Image Mask reads qwen21_edit_mask_alpha.png's alpha "
             "channel; Join Image With Alpha restores those RGBA channels before the "
             "native Qwen encoder sees image_1. The encoder composites transparent "
             "pixels over white for vision and sends the four-channel reference through "
             "the VAE. This is alpha-conditioned reference editing, not traditional "
             "locked-pixel or denoise-mask inpainting: this model exposes no separate "
             "mask input. The native source recipe stays 1024x1024, 25 Euler/simple "
             "steps at cfg 1.0, saving an RGBA PNG through the four-channel VAE. Expects "
             "qwen_image_2.1_bf16.safetensors in models/diffusion_models, "
             "qwen3vl_8b_bf16.safetensors in models/text_encoders, and "
             "qwen_image_2.1_vae_bf16.safetensors in models/vae. The prefix cache is "
             "off. The optional DGX cache node has lossless default storage, and its "
             "int8/int4 K/V compression is approximate. docs/MODELS.md records its "
             "limited native mode-equivalence scope, the exact 25-step T2I, "
             "masked-edit and ten-reference controls, and the classic three-step "
             "default/int8/int4 cache equivalence. These results do not extend past "
             "their named inputs, resolution, topology or FSDP scope. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "qwen_image_2.1_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image"}},
            {"key": "vae", "type": "VAELoader", "pos": (0, 3),
             "widgets": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            {"key": "source", "type": "LoadImage", "pos": (1, 3),
             "widgets": {"image": "qwen21_source.png"}},
            {"key": "mask", "type": "LoadImageMask", "pos": (1, 4),
             "widgets": {"image": "qwen21_edit_mask_alpha.png", "channel": "alpha"}},
            {"key": "rgba", "type": "JoinImageWithAlpha", "pos": (2, 3),
             "wire": {"image": ("source", 0), "alpha": ("mask", 0)}},
            {"key": "cond", "type": "TextEncodeQwenImage21", "pos": (2, 2),
             "widgets": {"prompt": "Edit only the alpha-matted region of image_1: "
                                      "replace the vase's painted decoration with a blue-and-gold "
                                      "botanical pattern, while retaining the visible studio, shape, "
                                      "lighting and all opaque source content.",
                         "negative_prompt": "blurry, low quality, watermark, text, logo", "resolution": 1024},
             "wire": {"clip": ("clip", 0), "vae": ("vae", 0),
                      "images.image_1": ("rgba", 0)}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (4, 2),
             "widgets": {"steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("cond", 0),
                      "negative": ("cond", 1), "latent_image": ("cond", 2)}},
            {"key": "dec", "type": "VAEDecode", "pos": (5, 2),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (6, 2),
             "widgets": {"filename_prefix": "dgx_monarch_qwen_image21_edit_mask_rgba"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-ernie-t2i": {
        "title": "Ernie Image text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Ernie Image text to image at 1024x1024, 20 steps at cfg 4.0, on the base "
             "checkpoint. topology=auto runs it on one Spark and asks for uly2 on a "
             "configured pair, the mode this family's hardware evidence covers; "
             "there is no resolution crossover row, so every canvas resolves the "
             "same way. Do not pin cfg2 here: this family pads no prompt, so a long "
             "positive and an empty negative cannot share one batched call, and "
             "cfg2 sends them to separate ranks, a path with no passing reference "
             "comparison for this family; auto never selects cfg2 for it. Under uly2 "
             "the joint image and text stream is sharded as one sequence, and the "
             "pad row an odd joint token count takes is dropped from attention: on "
             "2026-10-06 uly2 matched the single-GPU output exactly at 1024 and 1040, and on "
             "this graph. At 1024 square the image half is 64 by 64, an even 4096, "
             "so the text length decides whether a pad row is needed. The sampling "
             "shift of 3.0 lives in the model config, so this graph adds no model "
             "sampling node. The vendor recommends 50 steps; 20 is what the official "
             "ComfyUI template ships and what this family was accepted on, so start "
             "there. The prompt is a dense paragraph because the optional prompt "
             "enhancer, a second text model the official template switches in to "
             "expand short prompts, is not part of this graph. " + GATE + " Expects ernie-image.safetensors in "
             "models/diffusion_models, ministral-3-3b.safetensors in "
             "models/text_encoders, and flux2-vae.safetensors in models/vae: this "
             "family borrows the Flux 2 latent layout, 128 channels at /16 spatial, "
             "which only Empty Flux 2 Latent builds. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "ernie-image.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # The text encoder is Ministral 3 3B. Comfy routes it on the detected
            # weights, so the `type` value never reaches the loader's choice; it
            # stays flux2 only to match the official template.
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "ministral-3-3b.safetensors", "type": "flux2"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "High-fashion style summer outfit infographic "
                                 "featuring color-coordinated floating elements "
                                 "arranged in an elegant expanded circular "
                                 "composition. It includes a breathable straw hat, a "
                                 "sleeveless organic cotton top, a flowing pleated "
                                 "skirt, handcrafted leather sandals, and a woven palm "
                                 "leaf handbag. Exquisite annotations highlight fabric "
                                 "breathability, refreshing texture, moisture-wicking "
                                 "properties, and seasonal comfort. The color palette "
                                 "adopts warm neutral tones: ivory white, terracotta, "
                                 "sand, and soft tan. Subtle dynamic trajectories and "
                                 "flowing fabric swirls suggest a gentle summer breeze, "
                                 "while bright natural sunlight creates soft shadows "
                                 "and sun-kissed sheen, in a Mediterranean style."},
             "wire": {"clip": ("clip", 0)}},
            # An empty text encode, not Conditioning Zero Out: the official base
            # workflow wires a second empty encode, and that is the shape the
            # family was accepted on. Zero Out is the turbo workflow's pattern.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": ""}, "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyFlux2LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 4.0, "sampler_name": "euler",
                         "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "flux2-vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_ernie"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-zimage-t2i": {
        "title": "Z-Image text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto",
                         "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Z-Image text to image at 1536x1536 through the Turbo checkpoint: 8 "
             "steps at cfg 1.0 with res_multistep and the simple scheduler. Turbo "
             "is CFG distilled, so cfg stays at 1.0, and the Conditioning Zero Out "
             "negative gives the sampler a real unconditional; it steers nothing. "
             "1536x1536 is 2.36 MP, which keeps topology=auto on the uly2 row, the "
             "only scope this family has hardware evidence for, and it is the shape "
             "the 2026-08-11 run recorded (uly2 matched the DP2 reference at 1-step "
             "NRMS 0.000). Below 1.2 MP no zimage row applies, and auto takes the "
             "generic uly2 fallback instead. The official ComfyUI graph adds a shift "
             "3 model sampling patch that only restates the checkpoint default, so "
             "this template leaves it out; do not substitute Model Sampling SD3 "
             "(DGX Monarch), whose sd3 kind applies a 1000x timestep multiplier "
             "Z-Image already carries. Reference latents (the omni and edit path) "
             "and an effective caption attention mask both typed-refuse under "
             "sequence parallel, so keep the conditioning plain. The non-distilled "
             "z_image_bf16.safetensors is the higher-step alternative, 30 to 50 "
             "steps at cfg 3 to 5; keep it at or above 1.2 MP too, the band the "
             "2026-08-11 Z-Image uly2 evidence covers. " + GATE + " Expects "
             "z_image_turbo_bf16.safetensors in models/diffusion_models, "
             "qwen_3_4b.safetensors in models/text_encoders, and ae.safetensors, "
             "the Flux 1 autoencoder, in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "z_image_turbo_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # Z-Image uses the lumina2 CLIP target: ComfyUI's CLIPLoader type
            # list has no z_image entry, and ZImage subclasses Lumina2.
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen_3_4b.safetensors",
                         "type": "lumina2"}},
            # The sign text tests the family's two stated strengths,
            # photorealism and bilingual glyphs. 夜市 is Chinese for
            # "night market"; json.dump escapes it, so the generated JSON stays
            # ASCII.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": 'A weathered enamel sign hanging above a '
                                 'rain-slicked noodle shop at dusk, the sign '
                                 'reading "NIGHT MARKET" in crisp block letters '
                                 'with 夜市 beneath it, steam rising '
                                 'through warm lantern light, wet asphalt '
                                 'reflecting red and amber, shallow depth of '
                                 'field, 50mm photograph, fine grain.'},
             "wire": {"clip": ("clip", 0)}},
            # Turbo is CFG distilled and runs at cfg 1.0, so the negative only
            # has to be a real unconditional; zeroing the positive keeps both
            # conditionings the same length, which is what the family's cfg2
            # trim path wants if anyone raises cfg later.
            {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 2),
             "wire": {"conditioning": ("pos", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (2, 3),
             "widgets": {"width": 1536, "height": 1536, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 8, "cfg": 1.0,
                         "sampler_name": "res_multistep"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            # Z-Image has no VAE of its own; it decodes through the Flux 1
            # autoencoder, which Comfy-Org repacks alongside the checkpoint.
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_zimage"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-zimage-dct-t2i": {
        "title": "Z-Image DCT PixelSpace text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Z-Image DCT PixelSpace text to image at 1536x1536, 30 steps at cfg 4.0 "
             "with res_multistep. This surface is HW-scoped: on 2026-08-21 at 1536 "
             "square uly2 matched DP2 at one-step NRMS 0.000 and ring2 at 0.015, and "
             "equal-token cfg2 at 1024 square read 0.013. The steps and cfg come "
             "from the non-distilled latent Z-Image guidance because this model "
             "publishes no recipe of its own. The model decodes RGB patches through "
             "its own dec_net head after the sharded gather, so there is no VAE file "
             "to fetch: pixel_space is a built-in entry in the Load VAE list. The "
             "sampler tensor is raw RGB at the output size, so Empty Chroma Radiance "
             "Latent builds it, and the Empty Latent width and height are the "
             "decoded image size, not eight times smaller. topology=auto runs it on "
             "one Spark and splits it across the pair once a cluster.toml is "
             "configured; at 2.36 MP it lands on uly2, which this family's 30 "
             "attention heads tile and uly4 does not. Below 1.2 MP no zimage row "
             "applies, and auto takes the generic uly2 fallback instead. The "
             "checkpoint carries the sampling shift the official Z-Image graphs "
             "patch in, so this graph needs no model sampling node and has no shift "
             "widget. " + GATE +
             " Expects zeta-chroma-base-x0-pixel-no-dino.safetensors in "
             "models/diffusion_models and qwen_3_4b.safetensors in "
             "models/text_encoders. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "zeta-chroma-base-x0-pixel-no-dino.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # ZImagePixelSpace subclasses ZImage subclasses Lumina2, and the
            # checkpoint's cap_embedder.0.weight is 2560 wide, the Qwen3-4B hidden
            # size, so it takes the same text encoder as latent Z-Image on the
            # lumina2 CLIP target.
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
            # This reuses the latent template's bilingual signage prompt, so one
            # research sample prompt covers both Z-Image surfaces.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": 'A weathered enamel sign hanging above a '
                                 'rain-slicked noodle shop at dusk, the sign '
                                 'reading "NIGHT MARKET" in crisp block letters '
                                 'with 夜市 beneath it, steam rising '
                                 'through warm lantern light, wet asphalt '
                                 'reflecting red and amber, shallow depth of '
                                 'field, 50mm photograph, fine grain.'},
             "wire": {"clip": ("clip", 0)}},
            # cfg is above 1.0, so the sampler needs a real unconditional pass. The
            # non-distilled Z-Image template ships an empty encode here rather than
            # Conditioning Zero Out, and that is the family convention.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": ""}, "wire": {"clip": ("clip", 0)}},
            # ZImagePixelSpace's latent_format subclasses ChromaRadiance: 3 channels
            # at spacial_downscale_ratio 1. EmptySD3LatentImage would build a 16
            # channel /8 tensor that only the downscale_ratio_spacial rescale in
            # fix_empty_latent_channels turns into 1536x1536 pixels. Since 2026-10-05
            # the workers run that rescale (before, this rendered at 192x192) and the
            # driver's estimates read the rescaled shape, but the driver falls back
            # to the shape as built when comfy cannot name the checkpoint's config.
            # Empty Chroma Radiance Latent builds the right tensor directly, so the
            # driver prices the true 2.36 MP either way.
            {"key": "latent", "type": "EmptyChromaRadianceLatentImage", "pos": (2, 3),
             "widgets": {"width": 1536, "height": 1536, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 30, "cfg": 4.0, "sampler_name": "res_multistep",
                         "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            # pixel_space is a synthetic entry VAELoader.vae_list always appends; it
            # has no backing file, and load_vae builds an identity encode and decode.
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "pixel_space"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_zimage_dct"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-lens-t2i": {
        "title": "Lens text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "uly2", "mode": "auto",
                         "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Microsoft Lens text to image at 1344x1344, 20 steps at cfg 5.0 with "
             "euler and the simple scheduler. The cfg is real, so the negative "
             "encode is a live unconditional pass, and it must not be empty: the "
             "Lens tokenizer gives an empty string zero tokens, and the "
             "unconditional then fails. The latent is the Flux 2 "
             "layout, 128 channels at /16 spatial, which only Empty Flux 2 Latent "
             "builds. The topology is pinned to uly2, the scope this row measured: "
             "on 2026-07-28 uly2 matched the DP2 reference at NRMS 0.060 to 0.061 "
             "across 1328x1328 and 1344x1344, and on 2026-10-06 this graph's uly2 "
             "render matched the single-GPU output exactly. Auto reads this "
             "/16 latent at its true size and resolves the same uly2 row. The "
             "weights are bf16 only because fp16 produces NaNs, and padded ring and "
             "hybrid refuse instead of drifting unless an operator stamps the "
             "waiver, so stay on uly2 or cfg2. The stock ComfyUI graph for this "
             "family also carries a Flux model-sampling node and CFGNorm, neither of "
             "which takes a DGXM_MODEL: the built-in shift already covers a render "
             "this large and the measured run was plain CFG, so both are left out. "
             + GATE + " Expects "
             "lens_bf16.safetensors in models/diffusion_models, "
             "gpt_oss_20b_nvfp4.safetensors in models/text_encoders, a quantized "
             "encoder and the only published Comfy native form of it, and "
             "flux2-vae.safetensors in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "lens_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "gpt_oss_20b_nvfp4.safetensors",
                         "type": "lens"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A cluster of wild cosmos flowers swaying in gentle "
                                 "wind, crinkled soft petals and slender green stems, "
                                 "warm golden hour sunlight, natural field scenery, "
                                 "detailed floral texture, lifelike outdoor "
                                 "atmosphere"},
             "wire": {"clip": ("clip", 0)}},
            # Lens is not CFG distilled: cfg 5.0 runs a real unconditional pass.
            # The negative must not be empty: the Lens tokenizer emits zero
            # tokens for an empty string (no BOS floor), and the evaluated
            # unconditional then dies reshaping a 0-element tensor (measured
            # 2026-08-21). The row's uly2 evidence ran an asymmetric negative,
            # so the length mismatch is covered (docs/MODELS.md Lens row).
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low quality"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyFlux2LatentImage", "pos": (2, 3),
             "widgets": {"width": 1344, "height": 1344, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 5.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "flux2-vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_lens"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-omnigen2-t2i": {
        "title": "OmniGen2 text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto",
                         "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "OmniGen2 text to image at 1024x1024, 20 steps at cfg 5.0 with euler "
             "and the simple scheduler, and no shift node because the 2.6 sampling "
             "shift ships inside the checkpoint. The stock ComfyUI graph reaches "
             "these same numbers through a Dual CFG Guider whose second scale is "
             "reference image guidance, which does nothing without a reference "
             "image, so plain cfg 5.0 here is the same arithmetic and drops no "
             "value. Ulysses cannot split this family's 21 query heads, so ring is "
             "the only sequence-parallel topology: topology=auto picks ring2 at "
             "every resolution once a cluster.toml is configured, and one Spark runs "
             "the same graph unsharded. Ring2 shards one joined stream of text and "
             "image tokens and refuses an odd total, so both shipped prompts carry "
             "an even token count at 1024x1024. Change either prompt by one token "
             "and the render stops with a typed refusal naming the joined-token "
             "stream (docs/TROUBLESHOOTING.md #77): add or drop one short word to "
             "restore the count, or run topology=single. " +
             GATE + " Expects omnigen2_fp16.safetensors in "
             "models/diffusion_models, qwen_2.5_vl_fp16.safetensors in "
             "models/text_encoders, and ae.safetensors in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "omnigen2_fp16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen_2.5_vl_fp16.safetensors",
                         "type": "omnigen2"}},
            # Both prompts are even-length: the ring2 forward joins text and
            # image tokens into one stream and shard_seq refuses an odd total
            # (adapters/omnigen2.py, docs/TROUBLESHOOTING.md #77). 58 and 44 tokens
            # through the Omnigen2 tokenizer wrapper, 4096 image tokens at
            # 1024x1024, so both joined lengths are even.
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A cat with a crown lounging on a velvet throne, "
                                 "royal atmosphere, luxurious fabric texture, "
                                 "regal pose, detailed fur, ornate crown, "
                                 "dramatic lighting."},
             "wire": {"clip": ("clip", 0)}},
            # An empty negative is 25 tokens of chat template, which is odd and
            # refuses under ring2, so this family always ships real negative text.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "blurry, low quality, distorted, ugly, bad "
                                 "anatomy, deformed, poorly drawn."},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 20, "cfg": 5.0, "sampler_name": "euler",
                         "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_omnigen2"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-anima-t2i": {
        "title": "Anima text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Anima text to image at 1024x1024, 30 steps at cfg 4.0 with euler and the "
             "simple scheduler. Anima base is an untuned base model, so the positive "
             "prompt opens with the quality tag prefix the model card recommends and the "
             "negative carries the card's own list; a render without them looks "
             "flat. The CLIP loader type stays on stable_diffusion because ComfyUI "
             "picks this family from the detected Qwen3 0.6B encoder and lists no "
             "anima entry, and the sampling shift of 3.0 is stored in the checkpoint, so "
             "no model sampling node belongs in this graph. topology=auto runs it on one "
             "Spark and splits it across the pair once a cluster.toml is configured; an "
             "FSDP preset typed-refuses here because the llm_adapter never joins the "
             "sharded collective. " + GATE +
             " Expects anima-base-v1.0.safetensors in models/diffusion_models, "
             "qwen_3_06b_base.safetensors in models/text_encoders, and "
             "qwen_image_vae.safetensors in models/vae. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "anima-base-v1.0.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            # The type widget is inert for this family: comfy/sd.py routes Anima on the
            # detected Qwen3 0.6B encoder without testing clip_type, and the CLIPLoader
            # list holds no `anima` entry. stable_diffusion is what the official Anima
            # workflow ships.
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen_3_06b_base.safetensors",
                         "type": "stable_diffusion"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "masterpiece, best quality, score_7, safe, Anime "
                                 "monochrome cyberpunk front portrait, male figure, "
                                 "sleek skin with delicate mechanical lines, piercing "
                                 "glowing eyes, partial exposed metallic mecha "
                                 "components and light cables, sharp domineering cool "
                                 "style, textured anime brushwork, faint circuit "
                                 "background, high contrast chiaroscuro lighting, "
                                 "immersive cinematic shadows, ultra fine details, 8K "
                                 "high-def render, futuristic dystopian mood"},
             "wire": {"clip": ("clip", 0)}},
            # Anima runs true CFG at 4.0, so the negative is a real encode of the card's
            # recommended negative tags rather than a Conditioning Zero Out. Splitting
            # that pair across ranks is not on offer either: this family pads no
            # conditioning, so an asymmetric pair past the 512-token text length takes
            # the cfg-parallel typed refusal.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "worst quality, low quality, score_1, score_2, "
                                 "score_3, artist name, blurry, jpeg artifacts, "
                                 "chromatic aberration"},
             "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptyLatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024}},
            # euler and simple are the node defaults and the pair the official Anima
            # workflow ships; the card's own default, er_sde, is a legal one-widget swap
            # that changes no monarch contract.
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 30, "cfg": 4.0},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "qwen_image_vae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_anima"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-boogu-t2i": {
        "title": "Boogu text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "Boogu text to image at 1024x1024 through the base checkpoint: 25 steps "
             "at cfg 4.0, euler on the simple schedule, and no shift node because the "
             "3.16 sampling shift ships inside the checkpoint. Adding one here would "
             "not restate that value: Model Sampling SD3 takes the multiplier at its "
             "1000 default and this family declares 1.0, so the patch would hand the "
             "DiT a timestep a thousand times too large, and the render would come "
             "out as noise instead of refusing. The negative is an empty text "
             "encode on purpose, because Boogu turns an empty prompt into its own "
             "drop-prompt unconditional, and at cfg 4.0 that pass reaches the "
             "pixels. On a configured pair auto picks uly2 for this family at every "
             "canvas size; cfg2 stays an explicit preset and "
             "takes two prompts of different lengths by sending one to each rank. "
             "Under uly2 the conditional and unconditional text streams are sharded "
             "separately and their pad rows are dropped from attention: on "
             "2026-10-06 this graph's uly2 render matched the single-GPU output exactly. "
             + GATE + " Expects "
             "boogu_image_base_fp8_scaled.safetensors in models/diffusion_models, "
             "qwen3vl_8b_fp8_scaled.safetensors in models/text_encoders, and the Flux "
             "VAE, ae.safetensors, in models/vae: the text encoder is Qwen3-VL-8B, and "
             "a smaller Qwen VL file in its place fails on a hidden-size mismatch "
             "instead of giving a bad image. Boogu reuses the shared Flux "
             "autoencoder, which the family repack ships as "
             "flux1_vae_bf16.safetensors, so rename a VAE pulled from that repo to "
             "ae.safetensors or pick it from the list under the name it has. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "boogu_image_base_fp8_scaled.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
                         "type": "boogu"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "Abstract close-up portrait of a young man wearing a "
                                 "cream turtleneck, captured with severe horizontal "
                                 "motion blur and double exposure effect that distorts "
                                 "his facial features, rendered in an analog film grain "
                                 "style with muted earthy background tones, framed "
                                 "tightly on his face to emphasize the blur streaks "
                                 "across his eyes, nose, and lips while retaining the "
                                 "texture of the knit collar and soft ambient lighting."},
             "wire": {"clip": ("clip", 0)}},
            # Boogu's base checkpoint runs real CFG, so the negative is evaluated.
            # An empty text encode is the model-intended unconditional: comfy's
            # BooguTokenizer swaps in llama_template_drop when the prompt is blank
            # and no image is attached. Zeroing it out instead discards that branch.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": ""}, "wire": {"clip": ("clip", 0)}},
            {"key": "latent", "type": "EmptySD3LatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"steps": 25, "cfg": 4.0, "sampler_name": "euler",
                         "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_boogu"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-pixeldit-t2i": {
        "title": "PixelDiT text to image",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "PixelDiT text to image at 1024x1024, 30 steps at cfg 4.0 with er_sde "
             "and the simple scheduler. This family denoises pixels instead of a "
             "compressed latent, so the sampler tensor is raw RGB at the output "
             "size: Empty Chroma Radiance Latent builds it, the VAE loader is set "
             "to the built-in pixel_space pass-through, which needs no VAE file, "
             "and the sampling shift, 4.0 at 1024, is stored in the checkpoint, so "
             "no model sampling node belongs here. topology=auto resolves to uly2, "
             "the scope the hardware evidence covers: on 2026-08-11 uly2 and ring2 "
             "each matched a dp2 reference exactly across a 20-step 1024 render at "
             "cfg 4.0. cfg2 never applies because the text context is a fixed 300 "
             "tokens, and FSDP stays unclaimed. That acceptance ran euler at 20 "
             "steps; the 30-step er_sde recipe here is the vendor default, with a "
             "seed-driven noise sampler every rank shares. " + GATE +
             " Expects pixeldit_1300m_1024px_bf16.safetensors in "
             "models/diffusion_models and gemma_2_2b_it_elm_bf16.safetensors in "
             "models/text_encoders, with gemma_2_2b_it_elm_fp8_scaled.safetensors "
             "as the smaller text encoder swap; the weights are NSCLv1, "
             "non-commercial research or evaluation only. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name": "pixeldit_1300m_1024px_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "gemma_2_2b_it_elm_bf16.safetensors",
                         "type": "pixeldit"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "A surreal architectural scene featuring a woman "
                                 "in a flowing white dress walking away from the "
                                 "viewer through a narrow canyon of smooth, organic "
                                 "beige rock formations. The architecture resembles "
                                 "fluid sandstone or sculpted clay with undulating "
                                 "curves and soft edges. Bright sunlight streams "
                                 "from above, casting sharp shadows on the ground "
                                 "and illuminating the textured surfaces of the "
                                 "walls. The sky is a clear, deep blue visible at "
                                 "the top of the frame. The composition uses "
                                 "leading lines formed by the canyon walls to draw "
                                 "the eye toward the figure in the distance. "
                                 "High-quality photorealistic rendering with 8k "
                                 "resolution, cinematic lighting, dramatic contrast "
                                 "between light and shadow, and a sense of scale "
                                 "emphasizing the grandeur of the environment."},
             "wire": {"clip": ("clip", 0)}},
            # cfg 4.0 is a real CFG split, so the negative is a real encode. The
            # tokenizer pads both prompts to the same fixed 300-token context,
            # which is why this family carries no cfg-padding rule at all.
            {"key": "neg", "type": "CLIPTextEncode", "pos": (1, 3),
             "widgets": {"text": "low quality, worst quality, over-saturated, "
                                 "blurry, deformed, watermark"},
             "wire": {"clip": ("clip", 0)}},
            # PixelDiT's latent format is 3-channel RGB at spacial_downscale_ratio
            # 1, so width and height are the output pixels. 1024 gives a 64x64
            # patch stream of 4096 tokens, which splits evenly at sp=2.
            {"key": "latent", "type": "EmptyChromaRadianceLatentImage", "pos": (2, 3),
             "widgets": {"width": 1024, "height": 1024, "batch_size": 1}},
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"seed": 42, "steps": 30, "cfg": 4.0,
                         "sampler_name": "er_sde", "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("pos", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            # pixel_space is the synthetic identity VAE (see the Z-Image DCT graph).
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "pixel_space"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_pixeldit_t2i"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
    "dgx-monarch-pid-4k": {
        "title": "PiD pixel upscale 1024 to 4096",
        "nodes": [
            {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
             "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
            {"key": "note", "type": "Note", "pos": (0, 1), "note":
             "PiD pixel upscale: one 1024 source image redrawn at 4096x4096 in 4 "
             "distilled steps at cfg 1.0, with lcm and simple. topology=auto runs it "
             "on one Spark and resolves to uly2 across a configured pair, the "
             "topology the PiD leg was measured at; cfg-parallel never applies to "
             "this family because the text context is a fixed 300 tokens. Point Load "
             "Image at your own picture and rewrite the prompt to describe it, "
             "because the shipped filename and caption are placeholders; the chain "
             "after it scales to 1024x1024, encodes through the Flux 1 VAE, and lets "
             "PiD Conditioning attach that 16-channel latent with degrade_sigma 0. "
             "Keep cfg at 1.0: the negative is a Conditioning Zero Out of the raw "
             "text encode, so it carries no source latent, and comfy skips the "
             "unconditional pass only at exactly 1.0, so a higher cfg stops the "
             "render with 'PiD requires lq_latent'. The sampler latent is raw RGB at the "
             "full output size, a 65536-token patch stream at 4096x4096, so budget "
             "memory for it. " + GATE + " Expects "
             "pid_flux1_1024_to_4096_4step_bf16.safetensors in "
             "models/diffusion_models, gemma_2_2b_it_elm_bf16.safetensors in "
             "models/text_encoders, and ae.safetensors in models/vae for the source "
             "encode; the pixel stage decodes through the built-in pixel_space entry "
             "and needs no VAE file of its own. " + DERIVED},
            {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
             "widgets": {"unet_name":
                         "pid_flux1_1024_to_4096_4step_bf16.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "clip", "type": "CLIPLoader", "pos": (0, 2),
             "widgets": {"clip_name": "gemma_2_2b_it_elm_bf16.safetensors",
                         "type": "pixeldit"}},
            {"key": "pos", "type": "CLIPTextEncode", "pos": (1, 2),
             "widgets": {"text": "a weathered brass compass resting on a folded "
                                 "nautical chart, low afternoon light raking across "
                                 "the paper, fine engraved detail on the bezel, "
                                 "visible paper fibre and ink texture, sharp focus, "
                                 "photographic"},
             "wire": {"clip": ("clip", 0)}},
            # The negative is the raw encode zeroed out, not the PiD conditioning,
            # so it carries no lq_latent. That is why cfg is pinned to 1.0, the
            # only value at which comfy skips the unconditional pass. lcm is not a
            # cfg_pp sampler, so that skip holds (a cfg_pp sampler would evaluate
            # the negative at 1.0, as in the Krea 2 graph).
            {"key": "neg", "type": "ConditioningZeroOut", "pos": (2, 2),
             "wire": {"conditioning": ("pos", 0)}},
            {"key": "image", "type": "LoadImage", "pos": (0, 3),
             "widgets": {"image": "example.png", "upload": "image"}},
            {"key": "scale", "type": "ImageScale", "pos": (1, 3),
             "widgets": {"upscale_method": "lanczos", "width": 1024, "height": 1024,
                         "crop": "center"},
             "wire": {"image": ("image", 0)}},
            # The Flux 1 autoencoder, not the pixel stage: PiD's lq_proj wants a
            # 16-channel Flux1/SD3 latent and refuses any other channel count.
            {"key": "srcvae", "type": "VAELoader", "pos": (0, 4),
             "widgets": {"vae_name": "ae.safetensors"}},
            {"key": "enc", "type": "VAEEncode", "pos": (1, 4),
             "wire": {"pixels": ("scale", 0), "vae": ("srcvae", 0)}},
            {"key": "pid", "type": "PiDConditioning", "pos": (2, 3),
             "widgets": {"latent_format": "flux", "degrade_sigma": 0.0},
             "wire": {"positive": ("pos", 0), "latent": ("enc", 0)}},
            # PiD's latent format is PixelDiTPixel: 3 channels at full output
            # resolution, so the sampler latent is raw RGB and only the Chroma
            # Radiance empty latent builds it.
            {"key": "latent", "type": "EmptyChromaRadianceLatentImage", "pos": (2, 4),
             "widgets": {"width": 4096, "height": 4096, "batch_size": 1}},
            # Plain KSampler, not the custom sampler trio: it is numerically the
            # same graph as SamplerCustom + KSamplerSelect(lcm) +
            # BasicScheduler(simple, 4, 1.0) at add_noise true, and
            # DGXMonarchBasicScheduler refuses while the Init topology is `auto`
            # (auto resolves at the first render, after the sigmas are needed).
            {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
             "widgets": {"seed": 42, "steps": 4, "cfg": 1.0, "sampler_name": "lcm",
                         "scheduler": "simple"},
             "wire": {"model": ("unet", 0), "positive": ("pid", 0),
                      "negative": ("neg", 0), "latent_image": ("latent", 0)}},
            # pixel_space, the synthetic identity VAE, decodes the pixel stage.
            {"key": "vae", "type": "VAELoader", "pos": (3, 3),
             "widgets": {"vae_name": "pixel_space"}},
            {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
             "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
            {"key": "save", "type": "SaveImage", "pos": (5, 1),
             "widgets": {"filename_prefix": "dgx_monarch_pid_4k"},
             "wire": {"images": ("dec", 0)}},
        ],
    },
}
def _mage_flow_template(*, edit: bool, turbo: bool) -> dict:
    """Native Mage conditioning and sampling, with DGX owning the DiT only."""
    variant = "mage_flow" + ("_edit" if edit else "") + ("_turbo" if turbo else "")
    artifact = variant + ("_int8_convrot.safetensors" if turbo else "_bf16.safetensors")
    title = "Mage-Flow " + ("image editing" if edit else "text to image")
    title += " Turbo" if turbo else " quality"
    steps, cfg = (4, 1.0) if turbo else (30, 5.0)
    note = (f"{title}: {steps} Euler/simple steps at CFG {cfg:g}. "
            "Mage's native model configuration supplies shift 6.0 with multiplier 1; "
            "keep it unchanged. The native conditioning node supplies both branches "
            "and the 128-channel latent. ")
    if edit:
        note += ("Load a reference image and describe the edit you want. Width and "
                 "height 0 keep the reference size, floored to a multiple of 16; set "
                 "both for a different canvas. Add supporting images to the next "
                 "reference sockets. The node resizes VAE references to the output "
                 "canvas and keeps the same references in the negative branch. ")
    else:
        note += "Start at 1024x1024; dimensions must be multiples of 16. "
    evidence_scope = "one-reference Edit" if edit else "quality/Turbo T2I"
    note += (f"Expects {artifact}, mage_qwen3vl_4b_bf16.safetensors (CLIP type mage), "
             "and mage_flow_vae_bf16.safetensors from Comfy-Org/Mage-Flow. "
             "When you change quantization, pick the matching quality, edit or "
             "Turbo file. "
             f"See docs/MODELS.md for the named {evidence_scope} controls. The only FSDP "
             "first-use proof is Base T2I BF16, one step, with no LoRA; other flavors and "
             "quantized FSDP remain unmeasured. Other precision, multi-reference, "
             "LoRA-backed slab, and topology cases remain limited to their own records. "
             + GATE + " " + DERIVED)
    nodes = [
        {"key": "init", "type": "DGXMonarchInit", "pos": (0, 0),
         "widgets": {"topology": "auto", "mode": "auto", "auto_gate": "first_use"}},
        {"key": "note", "type": "Note", "pos": (0, 1), "note": note},
        {"key": "unet", "type": "DGXMonarchUNETLoader", "pos": (1, 0),
         "widgets": {"unet_name": artifact}, "wire": {"mesh": ("init", 0)}},
        {"key": "clip", "type": "CLIPLoader", "pos": (0, 3),
         "widgets": {"clip_name": "mage_qwen3vl_4b_bf16.safetensors", "type": "mage"}},
        {"key": "vae", "type": "VAELoader", "pos": (0, 4),
         "widgets": {"vae_name": "mage_flow_vae_bf16.safetensors"}},
    ]
    conditioning = {"clip": ("clip", 0), "vae": ("vae", 0)}
    if edit:
        nodes.append({"key": "reference", "type": "LoadImage", "pos": (1, 4),
                      "widgets": {"image": "reference.png", "upload": "image"}})
        conditioning["images.image_1"] = ("reference", 0)
    nodes.extend([
        {"key": "cond", "type": "TextEncodeMageFlowEdit", "pos": (2, 2),
         "widgets": {"prompt": ("Change the background to a sunlit garden. Preserve "
                                "the subject's identity and pose." if edit else
                                "A small red sailboat reflected in a calm alpine lake, "
                                "pine forest, crisp morning light, natural photograph."),
                     "negative_prompt": "", "width": 0 if edit else 1024,
                     "height": 0 if edit else 1024, "batch_size": 1},
         "wire": conditioning},
        {"key": "ks", "type": "DGXMonarchKSampler", "pos": (3, 1),
         "widgets": {"seed": 42, "steps": steps, "cfg": cfg,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
         "wire": {"model": ("unet", 0), "positive": ("cond", 0),
                  "negative": ("cond", 1), "latent_image": ("cond", 2)}},
        {"key": "dec", "type": "VAEDecode", "pos": (4, 1),
         "wire": {"samples": ("ks", 0), "vae": ("vae", 0)}},
        {"key": "save", "type": "SaveImage", "pos": (5, 1),
         "widgets": {"filename_prefix": "dgx_monarch_" + variant},
         "wire": {"images": ("dec", 0)}},
    ])
    return {"title": title, "nodes": nodes}


for _mage_edit in (False, True):
    for _mage_turbo in (False, True):
        _mage_name = "dgx-monarch-mage-flow-" + ("edit" if _mage_edit else "t2i")
        _mage_name += "-turbo" if _mage_turbo else ""
        TEMPLATES[_mage_name] = _mage_flow_template(edit=_mage_edit, turbo=_mage_turbo)


# dual-spark-split shares FLUX_STACK, whose latent is 1024. Override it here to
# 1536 so the split has enough tokens to be worth splitting.
TEMPLATES["dgx-monarch-dual-spark-split"]["nodes"] = [
    ({**n, "widgets": {"width": 1536, "height": 1536}}
     if n["key"] == "latent" else n)
    for n in TEMPLATES["dgx-monarch-dual-spark-split"]["nodes"]]


def _wan_animate2_quality_spec() -> dict:
    """The unaccelerated upstream Base recipe, not the Comfy fast-LoRA graph."""
    nodes = []
    for node in TEMPLATES["dgx-monarch-wan-animate2"]["nodes"]:
        if node["key"] in {"lora", "release"}:
            continue
        out = dict(node)
        if "widgets" in node:
            out["widgets"] = dict(node["widgets"])
        if "wire" in node:
            out["wire"] = dict(node["wire"])
        if node["key"] == "note":
            out["note"] = (
                "Wan-Animate 2 Base quality motion transfer: one 81-frame chunk, with "
                "a portrait 480x832 ImageScale (Upscale Image) by default. The 832x480 "
                "landscape canvas of the current upstream blueprint stays available: "
                "set the visible scale to 832x480 and use landscape source media. This "
                "is the unaccelerated upstream Base configuration, not the separate "
                "six-step Comfy LightX2V workflow: no LoRA, Euler/simple, 20 steps, CFG "
                "1.0, and the serialized DGX SD3 shift 5.0. Upstream's "
                "sample_guide_scale 0 disables CFG and keeps the conditional "
                "prediction, while Comfy CFG 0 selects the unconditional one, so CFG 1 "
                "is the matching conditional-only value. The raw driving video and "
                "reference image reach native Animate2 conditioning with both CLIP "
                "Vision encodes and separate appearance and pose prompts. The node "
                "resizes driving frames spatially, and the export uses the loaded "
                "driving video's FPS, so playback speed matches. To continue, duplicate "
                "this tail, wire the decoded frames to continue_motion and the prior "
                "video_frame_offset output to the next conditioner, then remove the next "
                "chunk's trim_image overlap before joining the chunks. Native MODEL "
                "cache and patch nodes cannot take DGXM_MODEL, so cache stays off. See "
                "docs/MODELS.md for the measured Base and Distilled BF16 recipes and the "
                "accelerated INT8+LightX2V six-step recipe. Unlisted settings, broader "
                "continuation, cache, first-use LoRA, slab/low-RSS, FSDP, ring, and "
                "background replacement have no passing reference comparison. This template ships slab "
                "and low-RSS off. " + NO_FIRST_USE_GATE + " Expects "
                "wan_animate_2_bf16.safetensors in models/diffusion_models, "
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors in models/text_encoders, "
                "clip_vision_h.safetensors in models/clip_vision, and "
                "Wan2_1_VAE_bf16.safetensors in models/vae. " + DERIVED
            )
        elif node["key"] == "shift":
            out["wire"] = {"model": ("unet", 0)}
        elif node["key"] == "unet":
            out["widgets"]["unet_name"] = "wan_animate_2_bf16.safetensors"
        elif node["key"] == "ks":
            out["widgets"].update({"steps": 20, "cfg": 1.0,
                                   "sampler_name": "euler", "scheduler": "simple"})
        elif node["key"] == "save":
            out["widgets"]["filename_prefix"] = "dgx_monarch_wan_animate2_base_quality"
        if node["key"] == "dec":
            out["wire"]["samples"] = ("trim", 0)
        nodes.append(out)
    return {"title": "Wan-Animate 2 Base quality motion transfer", "nodes": nodes}


TEMPLATES["dgx-monarch-wan-animate2-base-quality"] = _wan_animate2_quality_spec()


def _wan_animate2_memory_saver_spec() -> dict:
    """The explicit BF16 fallback that releases actors before VAE decode."""
    nodes = []
    for node in TEMPLATES["dgx-monarch-wan-animate2"]["nodes"]:
        out = dict(node)
        if "widgets" in node:
            out["widgets"] = dict(node["widgets"])
        if "wire" in node:
            out["wire"] = dict(node["wire"])
        if node["key"] == "note":
            out["note"] = (
                "Wan-Animate 2 memory-saving motion transfer: one 81-frame chunk at "
                "portrait 480x832 by default. This accelerated BF16+LightX2V recipe "
                "uses LCM, six steps, CFG 1.0, and the serialized DGX SD3 shift 5.0. It "
                "releases the client-owned actors after sampling and before VAE decode, "
                "which frees the DiT and LoRA memory and keeps the persistent Worker "
                "services and the driver VAE. Every later render reloads the model, so "
                "use this template only when lower memory use matters more than warm "
                "successive renders. A failed reset blocks decode. The reference "
                "prompt, pose prompt, both CLIP Vision encodes, driving-video FPS and "
                "native Animate2 conditioning match the standard template. Native "
                "MODEL cache and patch nodes cannot take DGXM_MODEL, so cache stays "
                "off. Slab and low-RSS are off. "
                + NO_FIRST_USE_GATE + " Expects "
                "wan_animate_2_bf16.safetensors in models/diffusion_models, "
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors in "
                "models/loras, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
                "models/text_encoders, clip_vision_h.safetensors in models/clip_vision, "
                "and Wan2_1_VAE_bf16.safetensors in models/vae. " + DERIVED
            )
        elif node["key"] == "unet":
            out["widgets"]["unet_name"] = "wan_animate_2_bf16.safetensors"
        elif node["key"] == "dec":
            out["pos"] = (7, 2)
            out["wire"] = {"samples": ("release", 1), "vae": ("vae", 0)}
        elif node["key"] == "video":
            out["pos"] = (8, 2)
        elif node["key"] == "save":
            out["pos"] = (9, 2)
            out["widgets"]["filename_prefix"] = (
                "video/dgx_monarch_wan_animate2_memory_saver"
            )
        nodes.append(out)
        if node["key"] == "trim":
            nodes.append({
                "key": "release", "type": "DGXMonarchClearVRAM", "pos": (6, 2),
                "widgets": {"level": "recycle", "include_driver": False},
                "wire": {"mesh": ("init", 0), "samples": ("trim", 0)},
            })
    return {"title": "Wan-Animate 2 memory-saving motion transfer", "nodes": nodes}


TEMPLATES["dgx-monarch-wan-animate2-memory-saver"] = _wan_animate2_memory_saver_spec()


def _wan_animate2_bf16_warm_spec() -> dict:
    """Warm BF16+LightX2V render that clears unused allocator buffers."""
    nodes = []
    for node in TEMPLATES["dgx-monarch-wan-animate2"]["nodes"]:
        out = dict(node)
        if "widgets" in node:
            out["widgets"] = dict(node["widgets"])
        if "wire" in node:
            out["wire"] = dict(node["wire"])
        if node["key"] == "note":
            out["note"] = (
                "Wan-Animate 2 BF16 warm motion transfer: one 81-frame chunk at portrait "
                "480x832 by default. This accelerated BF16+LightX2V recipe uses LCM, six "
                "steps, CFG 1.0, and the serialized DGX SD3 shift 5.0. Before native VAE "
                "decode, a soft cleanup clears unused worker allocator buffers and "
                "driver caches and keeps the actors and model weights resident for warm "
                "successive renders. The template uses ordinary weight and LoRA "
                "residency, with slab and low-RSS off. "
                + NO_FIRST_USE_GATE + " The standard INT8 template stays the default "
                "accelerated workflow. Native MODEL cache and patch nodes cannot take "
                "DGXM_MODEL, so cache stays off. Expects wan_animate_2_bf16.safetensors in "
                "models/diffusion_models, lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16"
                ".safetensors in models/loras, umt5_xxl_fp8_e4m3fn_scaled.safetensors in "
                "models/text_encoders, clip_vision_h.safetensors in models/clip_vision, and "
                "Wan2_1_VAE_bf16.safetensors in models/vae. " + DERIVED
            )
        elif node["key"] == "unet":
            out["widgets"]["unet_name"] = "wan_animate_2_bf16.safetensors"
        elif node["key"] == "dec":
            out["pos"] = (7, 2)
            out["wire"] = {"samples": ("soft_clear", 1), "vae": ("vae", 0)}
        elif node["key"] == "video":
            out["pos"] = (8, 2)
        elif node["key"] == "save":
            out["pos"] = (9, 2)
            out["widgets"]["filename_prefix"] = "video/dgx_monarch_wan_animate2_bf16_warm"
        nodes.append(out)
        if node["key"] == "trim":
            nodes.append({
                "key": "soft_clear", "type": "DGXMonarchClearVRAM", "pos": (6, 2),
                "widgets": {"level": "soft", "include_driver": True},
                "wire": {"mesh": ("init", 0), "samples": ("trim", 0)},
            })
    return {"title": "Wan-Animate 2 BF16 warm motion transfer", "nodes": nodes}


TEMPLATES["dgx-monarch-wan-animate2-bfloat16-warm"] = _wan_animate2_bf16_warm_spec()


# Sweep fixtures go to tests/fixtures/workflows/generated/, outside the template
# browser. Each one starts from the shipped spec of its family
# and changes only what its leg needs. Filenames run dgx-monarch-test-<name>.json,
# and a name ending in -lora is the family's LoRA variant
# (tests/test_example_workflows.py reads that suffix).

TESTING_SUBDIR = os.path.join("fixtures", "workflows", "generated")
TEST_LORA_NAME = "test_lora_placeholder.safetensors"
TEST_LORA_STRENGTH = 0.8

# Chroma shards the text stream under sequence parallel, where only Ulysses
# renders an odd count (see the comment on the shipped chroma negative).
# These two count 44 tokens each under comfy's chroma tokenizer (PixArt T5: no
# start token, one end token, no padding, so the stream is the tokenizer's own
# length), so they divide on every topology. The shipped pair counts 100 and
# 28: even too, but unequal, so it takes the cfg2 pad-and-mask path this equal
# pair skips. tools/count_t5_tokens.py counts either set.
CHROMA_EVEN_POSITIVE = (
    "A close-up photograph of a tiger's eye, speckled amber iris ringed by "
    "intricately detailed fur, natural light, subtle shadows, unfiltered "
    "amateur photography")
CHROMA_EVEN_NEGATIVE = (
    "low quality, blurry, watermark, text artifacts, oversaturated colors, "
    "deformed anatomy, noisy background, harsh flash, plastic skin, flat "
    "lighting, banding, heavy film grain")

# These are the synthetic inputs benchmark/sweep/media.py writes into comfy's
# input folder. A shipped template names demo media instead, which most installs
# lack.
TEST_IMAGE = "dgxm_test_portrait.png"
TEST_IMAGE_MASK = "dgxm_test_portrait_mask.png"
TEST_VIDEO = "dgxm_test_video.mp4"
TEST_VIDEO_MASK = "dgxm_test_mask.mp4"
TEST_AUDIO = "dgxm_test_audio.wav"

# The official two stage LTX 2.5 graph adds this three-step stage 2 schedule
# after the eight-step distilled one the shipped templates already carry.
LTX25_STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

# This prompt names what moves inside the kept region and what the model is
# free to build outside it, so a wrong mask shows up as a wrong picture, not a
# silent no-op.
LTX25_MASK_PROMPT = (
    "Use the provided start image as the first frame. The seated woman at the "
    "cafe table holds her pose and her face stays exactly as shown, lifting "
    "the cup slowly to her lips and setting it down again. Behind her the "
    "street fills with movement the frame does not yet show: a tram slides "
    "past left to right, pedestrians cross in both directions, shop awnings "
    "lift in the wind and late afternoon light rakes along the wet pavement. "
    "The camera holds a static medium shot throughout. Cup and saucer ring "
    "once, the tram bell sounds twice as it passes, footsteps and low crowd "
    "noise sit under a steady wind.")

# The vendor's multishot checklist: two shots inside five seconds, each cut
# named in prose, each new shot re-established, the woman named the same way
# twice, and the score said to carry over.
LTX25_MULTISHOT_PROMPT = (
    "A wide shot frames a rain slicked harbour front at dusk, sodium lamps "
    "doubling in the puddles along the quay. A woman in a mustard oilskin "
    "coat walks toward camera past stacked crab pots, hands deep in her "
    "pockets, while a fishing boat idles at the wall behind her. Low swell "
    "knocks against the stone, a slow piano figure plays under the scene, and "
    "rigging taps in the wind. A hard cut transitions to a medium close-up of "
    "the woman in the mustard oilskin coat under a lamp, rain beading on her "
    "shoulders and the light warm on one side of her face as she looks "
    "off-screen right; the piano figure continues across the cut and the "
    "swell drops back. She says, quietly, \"The tide has turned.\" She lifts "
    "her chin, and the camera holds as a gull crosses behind her and calls "
    "once.")

# The family's structured prompt shape, written so two beats sit where the two
# guides land: 1.7 s and 3.4 s.
H3_TWO_GUIDE_PROMPT = (
    "Realistic live-action cinematic look, workshop short: practical film "
    "photography, a bookbinder's bench under a north window, 40mm lens, "
    "shallow depth of field, cool daylight against a warm desk lamp, fine "
    "film grain.\n\n"
    "Scene overview: a pair of hands sets a stack of folded pages square on "
    "the bench, presses them flat, then lifts the finished block and turns it "
    "to the light.\n\n"
    "Storyboard (each shot its own scene, clean cuts):\n"
    "[0s-1.7s] Shot 1: high over the bench, the hands sliding folded pages "
    "into a neat stack, paper edges catching the window light.\n"
    "[1.7s-3.4s] Shot 2: the stack sits square under the press, one hand flat "
    "on top, the other easing the screw down.\n"
    "[3.4s-5s] Shot 3: the block lifts off the bench and turns toward the "
    "window, the spine folds opening and closing once.\n\n"
    "Camera: shot 1 holds a static overhead, shot 2 drifts in at slow speed, "
    "shot 3 tilts up with the block.\n\n"
    "Audio: paper sliding on paper throughout, the press screw creaking at "
    "1.8s, a soft wooden knock as the block sets down at 3.5s, a low room "
    "tone under all of it.\n\n"
    "No text, subtitles, logos or watermarks, no cartoon or CG rendering, "
    "keep the live-action texture.")

_MEDIA_PURPOSE = ("The media widgets name the synthetic files "
                  "benchmark/sweep/media.py writes into comfy's input folder.")
_LORA_PURPOSE = ("Load LoRA (DGX Monarch) connects the model loader to the sampler. "
                 f"Replace {TEST_LORA_NAME} with your LoRA file; the placeholder "
                 f"strength is {TEST_LORA_STRENGTH}.")


def _testing_note(base: str, purpose: str) -> str:
    return ("Testing-only workflow for the sweep harness; hidden from "
            f"the template browser. Based on {base}, with only the changes needed "
            f"for this test. {purpose}")


def _testing_spec(base: str, title: str, purpose: str, *,
                  widgets: dict[str, dict] | None = None,
                  rewire: dict[str, dict] | None = None,
                  add: tuple[dict, ...] = (),
                  lora_after: tuple[str, ...] = ()) -> dict:
    """The shipped spec of `base` with five edits: its note replaced, widget
    values overridden per node key, wires replaced per node key, nodes appended
    after the base graph, and a LoRA loader spliced into the model edge of each
    named UNET loader. The splice keeps the chain rule: the loader takes the
    model from the UNET loader, and every node that read that model, appended
    or not, reads the loader instead. An appended node is an ordinary spec node
    and names the base keys it reads."""
    base_nodes = TEMPLATES[base]["nodes"]
    overrides = widgets or {}
    rewires = rewire or {}
    lora_key = {unet: f"lora_{unet}" for unet in lora_after}
    nodes = []
    for node in base_nodes:
        out = dict(node)
        if node["type"] == "Note":
            out["note"] = _testing_note(base, purpose)
        patch = overrides.get(node["key"])
        if patch:
            out["widgets"] = {**node.get("widgets", {}), **patch}
        if node.get("wire"):
            out["wire"] = {name: (lora_key.get(src, src), slot)
                           for name, (src, slot) in node["wire"].items()}
        moved = rewires.get(node["key"])
        if moved:
            out["wire"] = {**out.get("wire", {}), **moved}
        nodes.append(out)
    for node in add:
        out = dict(node)
        if node.get("wire"):
            out["wire"] = {name: (lora_key.get(src, src), slot)
                           for name, (src, slot) in node["wire"].items()}
        nodes.append(out)
    # A free lane under the whole graph, so a spliced loader covers no node.
    lane = max(node["pos"][1] for node in nodes) + 1
    for offset, unet in enumerate(lora_after):
        column = next(n["pos"][0] for n in base_nodes if n["key"] == unet)
        nodes.append({"key": lora_key[unet], "type": "DGXMonarchLoraLoader",
                      "pos": (column, lane + offset),
                      "widgets": {"lora_name": TEST_LORA_NAME,
                                  "strength_model": TEST_LORA_STRENGTH},
                      "wire": {"model": (unet, 0)}})
    return {"title": title, "nodes": nodes}


TESTING_TEMPLATES = {
    "dgx-monarch-test-chroma-even": _testing_spec(
        "dgx-monarch-chroma-t2i", "Chroma even prompt pair",
        "Both prompts count 44 tokens: an even count the sequence-parallel "
        "shard divides with no pad row, and an equal pair the cfg path needs "
        "no mask for, so this graph reaches uly2 and uly2+fsdp with neither. "
        "The shipped pair is even as well, "
        "at 100 and 28, but unequal, so cfg2 pads and masks it.",
        widgets={"pos": {"text": CHROMA_EVEN_POSITIVE},
                 "neg": {"text": CHROMA_EVEN_NEGATIVE}}),
    "dgx-monarch-test-wandancer": _testing_spec(
        "dgx-monarch-wandancer", "WanDancer on synthetic media",
        _MEDIA_PURPOSE + " The audio is five seconds, so the global pass reads "
        "a high frame rate over one segment.",
        widgets={"image": {"image": TEST_IMAGE},
                 "audio": {"audio": TEST_AUDIO}}),
    "dgx-monarch-test-wan-animate2-int8": _testing_spec(
        "dgx-monarch-wan-animate2", "Wan-Animate 2 int8-convrot",
        "Exercises the vendor's native int8-convrot Base artifact; the stock "
        "conditioning and Base LoRA remain unchanged.",
        widgets={"unet": {"unet_name": "wan_animate_2_int8_convrot.safetensors"}}),
    "dgx-monarch-test-wan-animate2-distilled-int8": _testing_spec(
        "dgx-monarch-wan-animate2-distilled", "Wan-Animate 2 Distilled int8-convrot",
        "Exercises the vendor's native int8-convrot Distilled artifact with its "
        "no-LoRA recipe.",
        widgets={"unet": {"unet_name": "wan_animate_2_distill_int8_convrot.safetensors"}}),
    "dgx-monarch-test-wan-flowrvs": _testing_spec(
        "dgx-monarch-wan-flowrvs", "Wan FlowRVS on a synthetic clip",
        _MEDIA_PURPOSE + " The clip runs 81 frames, past the 17 this graph "
        "cuts.",
        widgets={"video": {"file": TEST_VIDEO}}),
    "dgx-monarch-test-wan-scail2": _testing_spec(
        "dgx-monarch-wan-scail2", "Wan SCAIL-2 on synthetic media",
        _MEDIA_PURPOSE + " All four inputs are synthetic. The shipped DPO LoRA "
        "stays: it is part of the recipe the row was measured on.",
        widgets={"video": {"file": TEST_VIDEO},
                 "maskvideo": {"file": TEST_VIDEO_MASK},
                 "image": {"image": TEST_IMAGE},
                 "maskimage": {"image": TEST_IMAGE_MASK}}),
    "dgx-monarch-test-flux2-lora": _testing_spec(
        "dgx-monarch-flux2-t2i", "Flux 2 with a LoRA slot", _LORA_PURPOSE,
        lora_after=("unet",)),
    "dgx-monarch-test-chroma-lora": _testing_spec(
        "dgx-monarch-chroma-t2i", "Chroma with a LoRA slot",
        _LORA_PURPOSE + " Both prompts count 44 tokens, so the "
        "sequence-parallel shard divides them with no pad row.",
        widgets={"pos": {"text": CHROMA_EVEN_POSITIVE},
                 "neg": {"text": CHROMA_EVEN_NEGATIVE}},
        lora_after=("unet",)),
    "dgx-monarch-test-krea2-lora": _testing_spec(
        "dgx-monarch-krea2-t2i", "Krea 2 with a LoRA slot", _LORA_PURPOSE,
        lora_after=("unet",)),
    "dgx-monarch-test-ideogram4-lora": _testing_spec(
        "dgx-monarch-ideogram4-t2i", "Ideogram 4 with a LoRA slot",
        _LORA_PURPOSE + " It patches the conditional model only; the "
        "unconditional checkpoint keeps its own weights.",
        lora_after=("unet",)),
    "dgx-monarch-test-zimage-lora": _testing_spec(
        "dgx-monarch-zimage-t2i", "Z-Image with a LoRA slot", _LORA_PURPOSE,
        lora_after=("unet",)),
    "dgx-monarch-test-kandinsky5-image-lora": _testing_spec(
        "dgx-monarch-kandinsky5-image", "Kandinsky 5 image with a LoRA slot",
        _LORA_PURPOSE, lora_after=("unet",)),
    "dgx-monarch-test-kandinsky5-video-lite-lora": _testing_spec(
        "dgx-monarch-kandinsky5-video-lite",
        "Kandinsky 5 Video Lite with a LoRA slot", _LORA_PURPOSE,
        lora_after=("unet",)),
    "dgx-monarch-test-ltx-t2v-lora": _testing_spec(
        "dgx-monarch-ltx-t2v", "LTX 2.3 text to video with a LoRA slot",
        _LORA_PURPOSE, lora_after=("unet",)),
    "dgx-monarch-test-ltx25-t2v-lora": _testing_spec(
        "dgx-monarch-ltx25-t2v", "LTX 2.5 text to video with a LoRA slot",
        _LORA_PURPOSE, lora_after=("unet",)),
    "dgx-monarch-test-minimax-h3-lora": _testing_spec(
        "dgx-monarch-minimax-h3-t2va", "MiniMax H3 with a LoRA slot",
        _LORA_PURPOSE, lora_after=("unet",)),
    "dgx-monarch-test-wan22-i2v-lora": _testing_spec(
        "dgx-monarch-wan22-i2v", "Wan 2.2 image to video with LoRA slots",
        _LORA_PURPOSE + " The mixture of experts takes two: one on the "
        "high-noise model and one on the low-noise model.",
        lora_after=("unet_high", "unet_low")),
    "dgx-monarch-test-qwen-image-lora": _testing_spec(
        "dgx-monarch-qwen-image-t2i", "Qwen-Image with a LoRA slot",
        _LORA_PURPOSE, lora_after=("unet",)),
    "dgx-monarch-test-wan22-t2v-lora": _testing_spec(
        "dgx-monarch-wan22-t2v", "Wan 2.2 text to video with LoRA slots",
        _LORA_PURPOSE + " The mixture of experts takes two: one on the "
        "high-noise model and one on the low-noise model.",
        lora_after=("high", "low")),

    # These five legs, added 2026-08-27, are ones the shipped set cannot
    # express: each is its family's shipped graph plus the nodes that leg needs.
    "dgx-monarch-test-ltx25-guide-mask": _testing_spec(
        "dgx-monarch-ltx25-i2v-guide", "LTX 2.5 guide with a spatial mask",
        "This graph puts a mask on Add Guide, so the guide frame steers only "
        "part of the picture. The second Load Image names the mask picture and Convert "
        "Image to Mask reads its red channel: white keeps the guide, black "
        "lets the model invent. A mask, or any strength under 1.0, turns on "
        "the per guide attention bias the base graph leaves off at strength "
        "1.0; uly2 re-expresses that bias and ring refuses it. " + _MEDIA_PURPOSE
        + " The guide and the mask are that pair, one soft shape on each, so a "
        "wrong mask shows up as a wrong picture rather than as nothing at all.",
        widgets={"pos": {"text": LTX25_MASK_PROMPT},
                 "image": {"image": TEST_IMAGE},
                 "simg": {"filename_prefix": "dgx_monarch_ltx25_guide_mask"},
                 "saud": {"filename_prefix":
                          "audio/dgx_monarch_ltx25_guide_mask"}},
        rewire={"guide": {"attention_mask": ("mask", 0)}},
        add=(
            # Load Image's own MASK output is the inverted alpha channel, the
            # wrong sign here, so the red channel is read instead.
            {"key": "maskimg", "type": "LoadImage", "pos": (0, 6),
             "widgets": {"image": TEST_IMAGE_MASK, "upload": "image"}},
            {"key": "mask", "type": "ImageToMask", "pos": (0, 7),
             "widgets": {"channel": "red"},
             "wire": {"image": ("maskimg", 0)}},
        )),
    "dgx-monarch-test-ltx25-multishot": _testing_spec(
        "dgx-monarch-ltx25-t2v", "LTX 2.5 multishot text to video with sound",
        "One render cuts between shots and keeps the same people, room, light "
        "and voice across the cuts. There is no multishot node: the "
        "whole feature lives in the prompt, so the leg needs a template that "
        "carries one. The vendor's rules, which this prompt follows: one "
        "chronological paragraph rather than a shot list, each transition "
        "named in plain words, the new shot re-established after every cut, "
        "the same words for a recurring person, and the soundtrack described "
        "across the cuts. Two to four shots per render is the working range. "
        "Shape, schedule and sampler stay the base graph's.",
        widgets={"pos": {"text": LTX25_MULTISHOT_PROMPT},
                 "simg": {"filename_prefix": "dgx_monarch_ltx25_multishot"},
                 "saud": {"filename_prefix":
                          "audio/dgx_monarch_ltx25_multishot"}}),
    "dgx-monarch-test-ltx25-two-stage": _testing_spec(
        "dgx-monarch-ltx25-t2v", "LTX 2.5 two stage text to video with sound",
        "The shape the official template ships and the base graph leaves out: "
        "sample at half the canvas, double the video latent with the spatial "
        "upscaler, then sample three more steps at the full canvas. The 22B "
        "DiT runs twice, so the render costs about twice the base and recovers "
        "the detail the base gives up. Separate AV Latent splits the packed "
        "pair, because the upscaler takes a plain video latent; the audio half "
        "skips the upscaler and re-enters stage 2 beside the upscaled video, "
        "which re-noises and re-samples both. Stage 2 runs the shorter "
        "schedule, "
        "which starts below 1.0 and so refines instead of building from "
        "noise. 640x352 doubles to 1280x704, and both stages keep an even row "
        "count at world 2. Adds "
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors in "
        "models/latent_upscale_models.",
        widgets={"vlatent": {"width": 640, "height": 352},
                 "simg": {"filename_prefix": "dgx_monarch_ltx25_two_stage"},
                 "saud": {"filename_prefix":
                          "audio/dgx_monarch_ltx25_two_stage"}},
        rewire={"vdec": {"samples": ("sep2", 0)},
                "adec": {"samples": ("sep2", 1)}},
        add=(
            {"key": "upmodel", "type": "LatentUpscaleModelLoader",
             "pos": (0, 6),
             "widgets": {"model_name":
                         "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0"
                         ".safetensors"}},
            # Slot 0 is the sampled latent, which is what the official graph
            # upscales, not the denoised slot.
            {"key": "sep1", "type": "LTXVSeparateAVLatent", "pos": (6, 0),
             "wire": {"av_latent": ("sc", 0)}},
            {"key": "up", "type": "LTXVLatentUpsampler", "pos": (6, 1),
             "wire": {"samples": ("sep1", 0), "upscale_model": ("upmodel", 0),
                      "vae": ("vvae", 0)}},
            {"key": "latent2", "type": "LTXVConcatAVLatent", "pos": (6, 2),
             "wire": {"video_latent": ("up", 0),
                      "audio_latent": ("sep1", 1)}},
            {"key": "sigmas2", "type": "ManualSigmas", "pos": (6, 3),
             "widgets": {"sigmas": LTX25_STAGE2_SIGMAS}},
            {"key": "sampler2", "type": "KSamplerSelect", "pos": (6, 4),
             "widgets": {"sampler_name": "euler_ancestral"}},
            {"key": "noise2", "type": "RandomNoise", "pos": (6, 5),
             "widgets": {"noise_seed": 42}},
            # A second guider on the same model and the same conditioning,
            # which is how the official graph draws it.
            {"key": "guider2", "type": "DGXMonarchCFGGuider", "pos": (7, 0),
             "widgets": {"cfg": 1.0},
             "wire": {"model": ("unet", 0), "positive": ("ltxc", 0),
                      "negative": ("ltxc", 1)}},
            {"key": "sc2", "type": "DGXMonarchSamplerCustom", "pos": (7, 1),
             "wire": {"noise": ("noise2", 0), "guider": ("guider2", 0),
                      "sampler": ("sampler2", 0), "sigmas": ("sigmas2", 0),
                      "latent_image": ("latent2", 0)}},
            {"key": "sep2", "type": "LTXVSeparateAVLatent", "pos": (8, 0),
             "wire": {"av_latent": ("sc2", 0)}},
        )),
    "dgx-monarch-test-minimax-h3-guides": _testing_spec(
        "dgx-monarch-minimax-h3-guide",
        "MiniMax H3 with two guides in sequence",
        "A second Add Guide reads the first one's positive output, so two "
        "stills anchor at two frames at once. Add Guide appends to the "
        "keyframe list already on the conditioning, so a third would hold "
        "three, and each guide needs its own Load Image and its own "
        "frame_idx. 41 and 82 are 1.7 s and 3.4 s into a 124 frame clip at 24 "
        "fps, and the prompt puts a described beat at each. Keep every guide "
        "to one frame: on 2026-08-14 the video VAE encode before the guide "
        "took 13.6 GiB for one frame at this canvas and over 94 GiB for five, "
        "and two chained guides run that encode twice.",
        widgets={"cond": {"prompt": H3_TWO_GUIDE_PROMPT},
                 "guide": {"frame_idx": 41},
                 "simg": {"filename_prefix":
                          "dgx_monarch_minimax_h3_guides"},
                 "saud": {"filename_prefix":
                          "audio/dgx_monarch_minimax_h3_guides"}},
        rewire={"ks": {"positive": ("guide2", 0)}},
        add=(
            {"key": "image2", "type": "LoadImage", "pos": (0, 6),
             "widgets": {"image": "example.png", "upload": "image"}},
            # Both guides read the same latent: it is the canvas they resolve
            # frame_idx against, not a value they consume.
            {"key": "guide2", "type": "MiniMaxH3AddGuide", "pos": (2, 5),
             "widgets": {"frame_idx": 82},
             "wire": {"positive": ("guide", 0), "latent": ("latent", 0),
                      "vae": ("vvae", 0), "image": ("image2", 0)}},
        )),
    "dgx-monarch-test-hunyuan-image-refiner": _testing_spec(
        "dgx-monarch-hunyuan-image",
        "HunyuanImage 2.1 text to image with the refiner",
        "This graph adds the refiner tail the base note describes. The refiner is a "
        "separate checkpoint with its own VAE and its own latent layout, so "
        "it cannot resample the first stage's latent: the pixels come back "
        "through the 2.1 VAE, go out again through the refiner VAE, and "
        "Hunyuan Latent Refiner writes that encode onto both conditionings "
        "while handing back a fresh 32 channel canvas of zeros for the "
        "refiner to fill. noise_augmentation says how much noise the refiner "
        "is told the input carries, and 0.10 is the node default. 4 steps at "
        "cfg 1.0. Both checkpoints take the mesh from the same Init, so the "
        "pair holds two 17B class models at once and the capacity preflight "
        "prices both. Adds hunyuanimage2.1_refiner_fp8_e4m3fn.safetensors in "
        "models/diffusion_models and "
        "hunyuan_image_refiner_vae_fp16.safetensors in models/vae.",
        widgets={"save": {"filename_prefix":
                          "dgx_monarch_hunyuan_image_refiner"}},
        rewire={"save": {"images": ("rdec", 0)}},
        add=(
            # Second checkpoint, same mesh. A graph that samples the base
            # model twice is not a refiner run.
            {"key": "runet", "type": "DGXMonarchUNETLoader", "pos": (1, 1),
             "widgets": {"unet_name":
                         "hunyuanimage2.1_refiner_fp8_e4m3fn.safetensors"},
             "wire": {"mesh": ("init", 0)}},
            {"key": "rvae", "type": "VAELoader", "pos": (0, 3),
             "widgets": {"vae_name":
                         "hunyuan_image_refiner_vae_fp16.safetensors"}},
            {"key": "renc", "type": "VAEEncode", "pos": (6, 1),
             "wire": {"pixels": ("dec", 0), "vae": ("rvae", 0)}},
            {"key": "rlat", "type": "HunyuanRefinerLatent", "pos": (6, 2),
             "widgets": {"noise_augmentation": 0.10},
             "wire": {"positive": ("pos", 0), "negative": ("neg", 0),
                      "latent": ("renc", 0)}},
            {"key": "rks", "type": "DGXMonarchKSampler", "pos": (7, 1),
             "widgets": {"steps": 4, "cfg": 1.0},
             "wire": {"model": ("runet", 0), "positive": ("rlat", 0),
                      "negative": ("rlat", 1), "latent_image": ("rlat", 2)}},
            {"key": "rdec", "type": "VAEDecode", "pos": (8, 1),
             "wire": {"samples": ("rks", 0), "vae": ("rvae", 0)}},
        )),
}


def _template_membership_drift(out_dir: str, names, cards: bool = True) -> list[str]:
    # Cards are the browser's tile art. The testing-only set never reaches the
    # browser, so it declares no jpg and any jpg found there is drift.
    expected = {
        ".json": {f"{name}.json" for name in names},
        ".jpg": {f"{name}.jpg" for name in names} if cards else set(),
    }
    try:
        entries = set(os.listdir(out_dir))
    except FileNotFoundError:
        return [f"{out_dir} (missing directory)"]
    drift = []
    for suffix, expected_names in expected.items():
        actual_names = {name for name in entries if name.endswith(suffix)}
        drift.extend(
            f"{os.path.join(out_dir, name)} (missing)"
            for name in sorted(expected_names - actual_names)
        )
        drift.extend(
            f"{os.path.join(out_dir, name)} (unexpected)"
            for name in sorted(actual_names - expected_names)
        )
    return drift


@_with_comfy_stubs
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail when committed templates differ; do not write")
    parser.add_argument(
        "--output-dir",
        help="template directory (default: repository example_workflows)",
    )
    parser.add_argument(
        "--fixtures-dir",
        help=("fixture directory (default: tests/fixtures/workflows/generated; "
              "with --output-dir, its fixtures/workflows/generated subdirectory)"),
    )
    args = parser.parse_args(argv)
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "example_workflows"
    )
    testing_dir = args.fixtures_dir or os.path.join(
        args.output_dir if args.output_dir else os.path.join(
            os.path.dirname(__file__), "..", "tests"),
        TESTING_SUBDIR,
    )
    sets = ((out_dir, TEMPLATES, True), (testing_dir, TESTING_TEMPLATES, False))
    drift = []
    for directory, specs, cards in sets:
        if args.check:
            drift.extend(_template_membership_drift(directory, specs, cards))
        else:
            os.makedirs(directory, exist_ok=True)
        for name, spec in specs.items():
            path = os.path.join(directory, f"{name}.json")
            generated = build(spec)
            if args.check:
                try:
                    with open(path) as f:
                        committed = json.load(f)
                except (OSError, ValueError):
                    committed = None
                if committed != generated:
                    drift.append(path)
                continue
            with open(path, "w") as f:
                json.dump(generated, f, indent=1)
            print(f"wrote {path}")
    if drift:
        print(
            "template drift; regenerate JSON and reconcile declared JSON/JPEG assets:\n  "
            + "\n  ".join(drift),
            file=sys.stderr,
        )
        return 1
    if args.check:
        print(f"templates in sync ({len(TEMPLATES)} shipped, "
              f"{len(TESTING_TEMPLATES)} testing-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
