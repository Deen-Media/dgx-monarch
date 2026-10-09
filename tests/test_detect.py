"""Checkpoint family and quant sniffing from synthetic headers, and real ones where present."""
import json
import math
import struct
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dgx_monarch.adapters.detect import (
    _SIGNATURES,
    CheckpointSniffError,
    _read_sniff,
    _sniff,
    _strip_prefix,
    detect_family_from_header,
    detect_family_from_keys,
    detect_quant_from_header,
    fsdp_lora_admission_property,
    is_zimage_l2p_from_keys,
    nearest_signatures,
    sniff_checkpoint,
    sniff_checkpoint_for_topology,
    sniff_fsdp_launch_quant,
    sniff_metadata_keys,
    sniff_nearest_signatures,
    sniff_zimage_l2p,
)


def _write_safetensors_header(path: Path, tensors: dict[str, dict]):
    header = json.dumps(tensors).encode()
    data_bytes = max(
        (info["data_offsets"][1]
         for name, info in tensors.items()
         if name != "__metadata__" and isinstance(info, dict)
         and isinstance(info.get("data_offsets"), list)
         and len(info["data_offsets"]) == 2),
        default=0,
    )
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        # The parser never reads these bytes, but it checks that every
        # advertised range lies inside the file.
        f.write(b"\0" * data_bytes)


def _write_dtype_checkpoint(path: Path, dtypes: list[str]) -> None:
    widths = {
        "BF16": 2,
        "F16": 2,
        "F8_E4M3": 1,
        "I8": 1,
        "F32": 4,
        "F64": 8,
        "I64": 8,
    }
    offset = 0
    tensors = {}
    for index, dtype in enumerate(dtypes):
        width = widths[dtype]
        tensors[f"weight_{index}"] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + width],
        }
        offset += width
    _write_safetensors_header(path, tensors)


FAMILY_KEYS = {
    "krea2": ["txtfusion.projector.weight", "blocks.0.attn.wq.weight"],
    "ideogram4": ["llm_cond_proj.weight", "layers.0.attention.qkv.weight"],
    "chroma": ["distilled_guidance_layer.in_proj.weight", "double_blocks.0.img_attn.qkv.weight"],
    "ltx": ["adaln_single.linear.weight", "transformer_blocks.0.attn1.to_q.weight",
            "patchify_proj.weight"],
    "wan": ["text_embedding.0.weight", "blocks.0.self_attn.q.weight"],
    "minimax_h3": ["video_patch_proj.weight", "audio_patch_proj.weight",
                   "blocks.0.attn.qkv_proj.weight"],
}


# The full H3 parameter namespace, under the comfy wrapper prefix so the row is
# also proved through _strip_prefix. The entry that matters is the top-level
# token_refiner.: hunyuan's three rows need the nested
# txt_in.individual_token_refiner., which it does not satisfy. Likewise blocks.
# and mlp.fc2. must not reach omnigen2 (bare layers.) or ernie (mlp.linear_fc2.).
MINIMAX_H3_KEYS = [
    "model.diffusion_model.video_patch_proj.weight",
    "model.diffusion_model.audio_patch_proj.weight",
    "model.diffusion_model.condition_proj.weight",
    "model.diffusion_model.token_refiner.blocks.0.attn.qkv_proj.weight",
    "model.diffusion_model.token_refiner.final_norm.weight",
    "model.diffusion_model.blocks.0.attn.qkv_proj.weight",
    "model.diffusion_model.blocks.0.mlp.fc2.weight",
    "model.diffusion_model.blocks.0.adaln_proj.linear.weight",
    "model.diffusion_model.final_layer.video_out.weight",
    "model.diffusion_model.final_layer.audio_out.weight",
    "model.diffusion_model.rope.inv_freq",
]


def test_minimax_h3_full_namespace_claims_no_other_family():
    assert detect_family_from_keys(MINIMAX_H3_KEYS) == "minimax_h3"


def test_minimax_h3_top_level_token_refiner_is_not_hunyuan():
    """H3 without its two discriminating projections must fall through to
    unknown, never to hunyuan on the refiner name alone."""
    keys = [k for k in MINIMAX_H3_KEYS if "patch_proj" not in k]
    assert detect_family_from_keys(keys) == "unknown"


HUNYUAN_REFINER_KEYS = [
    "model.model.txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight",
    "model.model.time_r_in.in_layer.weight",
    "model.model.double_blocks.0.img_attn.qkv.weight",
    "model.model.single_blocks.0.linear1.weight",
]


@pytest.mark.parametrize("family", sorted(FAMILY_KEYS))
def test_family_signatures(family):
    assert detect_family_from_keys(FAMILY_KEYS[family]) == family


def test_wan_animate2_uses_only_the_native_config_discriminator(tmp_path: Path):
    """The official file is ordinary Wan i2v by tensor names.

    `patch_embedding.weight` has the documented 36-channel shape, but that is
    loader information rather than a family signature: only Comfy's preserved
    config JSON may claim the Animate2 adapter.
    """
    path = tmp_path / "animate2.safetensors"
    _write_safetensors_header(path, {
        "patch_embedding.weight": {"dtype": "BF16", "shape": [5120, 36, 1, 2, 2], "data_offsets": [0, 1474560]},
        "text_embedding.0.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [1474560, 1474562]},
        "blocks.0.self_attn.q.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [1474562, 1474564]},
        "__metadata__": {"config": '{"transformer":{"model_type":"animate2"}}'},
    })
    assert sniff_checkpoint(path) == ("wan_animate2", "bf16")


@pytest.mark.parametrize("config", [None, "", "not json", "[]", "{}", '{"transformer":[]}', '{"transformer":{"model_type":"i2v"}}'])
def test_wan_animate2_does_not_invent_a_tensor_signature(tmp_path: Path, config):
    path = tmp_path / "ordinary_i2v.safetensors"
    metadata = {} if config is None else {"config": config}
    _write_safetensors_header(path, {
        "patch_embedding.weight": {"dtype": "BF16", "shape": [5120, 36, 1, 2, 2], "data_offsets": [0, 1474560]},
        "text_embedding.0.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [1474560, 1474562]},
        "blocks.0.self_attn.q.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [1474562, 1474564]},
        "__metadata__": metadata,
    })
    assert sniff_checkpoint(path)[0] == "wan"


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model.", "diffusion_model."])
def test_prefix_stripping(prefix):
    keys = [prefix + k for k in FAMILY_KEYS["krea2"]]
    assert detect_family_from_keys(keys) == "krea2"


def test_nested_model_prefix_is_normalized_as_one_comfy_wrapper():
    key = "model.model.txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight"
    assert _strip_prefix(key) == "txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight"


def test_hunyuan_image_refiner_precedes_generic_longcat_signature():
    assert detect_family_from_keys(HUNYUAN_REFINER_KEYS) == "hunyuan"


def test_time_r_without_token_refiner_does_not_claim_hunyuan():
    keys = [
        "model.model.time_r_in.in_layer.weight",
        "model.model.double_blocks.0.img_attn.qkv.weight",
        "model.model.single_blocks.0.linear1.weight",
    ]
    assert detect_family_from_keys(keys) == "longcat"


def test_hunyuan_video_10_token_refiner_alone_remains_outside_launch_set():
    keys = [
        "model.model.txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight",
        "model.model.double_blocks.0.img_attn.qkv.weight",
        "model.model.single_blocks.0.linear1.weight",
    ]
    assert detect_family_from_keys(keys) == "longcat"


# Two spellings of one architecture, both read on 2026-09-09 off staged
# CogVideoX 1.5 5B exports. The ComfyUI layout flattens the time and OFS
# embeddings and names its blocks `blocks.`; a file straight out of diffusers
# keeps the dotted embedding names and `transformer_blocks.`. The OFS rows are
# the I2V and inpaint surfaces; a T2V export of either layout carries neither
# (docs/TROUBLESHOOTING.md #102).
COGVIDEO_COMFY_KEYS = [
    "patch_embed.proj.weight",
    "patch_embed.text_proj.weight",
    "time_embedding_linear_1.weight",
    "time_embedding_linear_2.weight",
    "ofs_embedding_linear_1.weight",
    "ofs_embedding_linear_2.weight",
    "blocks.0.norm1.linear.weight",
    "blocks.0.attn_out.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
]

COGVIDEO_DIFFUSERS_KEYS = [
    "patch_embed.proj.weight",
    "patch_embed.text_proj.weight",
    "time_embedding.linear_1.weight",
    "time_embedding.linear_2.weight",
    "ofs_embedding.linear_1.weight",
    "ofs_embedding.linear_2.weight",
    "transformer_blocks.0.norm1.linear.weight",
    "transformer_blocks.0.attn1.to_q.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
]


def test_cogvideox_comfy_layout_reads_cogvideo():
    assert detect_family_from_keys(COGVIDEO_COMFY_KEYS) == "cogvideo"


def test_cogvideox_diffusers_layout_reads_cogvideo():
    """The 2026-09-09 miss: the staged I2V export is in the diffusers layout,
    whose time embedding is `time_embedding.linear_1`, so the flattened needle
    alone read it as unknown."""
    assert detect_family_from_keys(COGVIDEO_DIFFUSERS_KEYS) == "cogvideo"


def test_cogvideox_diffusers_header_sniffs_cogvideo(tmp_path: Path):
    """The same reading through the header parser a render uses."""
    path = tmp_path / "cogvideox_1_5_i2v_diffusers.safetensors"
    tensors = {}
    offset = 0
    for name in COGVIDEO_DIFFUSERS_KEYS:
        tensors[name] = {"dtype": "BF16", "shape": [1],
                         "data_offsets": [offset, offset + 2]}
        offset += 2
    _write_safetensors_header(path, tensors)
    assert sniff_checkpoint(path) == ("cogvideo", "bf16")


def test_cogvideox_text_projection_alone_is_not_enough():
    """Neither row may fall back on the one path they share. A header holding
    `patch_embed.text_proj.` and no time embedding of either spelling still
    reads unknown, and reports the 1 of 2 near miss quoted in
    docs/TROUBLESHOOTING.md #102."""
    keys = [k for k in COGVIDEO_DIFFUSERS_KEYS if "time_embedding" not in k]
    assert detect_family_from_keys(keys) == "unknown"
    best = nearest_signatures(keys)[0]
    assert (best.family, best.present, best.total) == ("cogvideo", 1, 2)


def test_unknown_family():
    assert detect_family_from_keys(["some.random.weight"]) == "unknown"


def test_pixart_layout_is_not_ltx():
    # diffusers-layout PixArt carries adaln_single + transformer_blocks but no
    # patchify_proj, so it must not sniff as ltx.
    keys = ["adaln_single.linear.weight", "transformer_blocks.0.attn1.to_q.weight",
            "pos_embed.proj.weight"]
    assert detect_family_from_keys(keys) == "unknown"


def test_mid_name_substring_does_not_match():
    # "xtxtfusion." contains "txtfusion." mid-name; segment matching rejects it.
    assert detect_family_from_keys(["xtxtfusion.projector.weight"]) == "unknown"


def test_dotless_needle_decoy_does_not_match():
    # zimage's "x_pad_token" is the one needle with no trailing dot, so a key
    # that only starts with it ("x_pad_tokenizer...") must not satisfy the
    # segment match (the `_has_segment` docstring).
    keys = ["cap_embedder.weight", "x_pad_tokenizer.weight"]
    assert detect_family_from_keys(keys) == "unknown"


def test_dotless_needle_leaf_match():
    # A complete "x_pad_token" leaf segment beside cap_embedder. does satisfy
    # the zimage signature.
    keys = ["cap_embedder.weight", "x_pad_token"]
    assert detect_family_from_keys(keys) == "zimage"


def test_non_safetensors_raises_typed(tmp_path: Path):
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"\x80\x04\x95" + b"\xff" * 64)  # pickle-ish garbage
    with pytest.raises(CheckpointSniffError):
        sniff_checkpoint(path)


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        (["BF16"], "bf16"),
        (["F32"], "fp32"),
        (["BF16", "F32"], "fp32"),  # one element each: fp32 is 2/3 of the bytes
        (["F16", "F32"], "fp32"),  # fp16 cores are admitted uniform only
        (["F8_E4M3", "F32"], "fp8"),
        (["I8", "F32"], "int8"),
        (["F64"], "unknown"),
        (["BF16", "I64"], "unknown"),
        ([], "unknown"),
    ],
)
def test_fsdp_launch_quant_requires_literal_bf16(
    tmp_path: Path,
    dtypes,
    expected,
):
    path = tmp_path / "model.safetensors"
    _write_dtype_checkpoint(path, dtypes)
    assert sniff_fsdp_launch_quant(path) == expected


@pytest.mark.parametrize(("core_elements", "expected"), [
    (1000, "bf16"),  # one fp32 element beside 1000 bf16: inside the 5% ceiling
    (30, "fp32"),  # 4 of 64 bytes: over it
])
def test_fsdp_launch_quant_admits_fp32_islands_inside_the_ceiling(
    tmp_path: Path, core_elements, expected,
):
    path = tmp_path / "model.safetensors"
    core_bytes = 2 * core_elements
    _write_safetensors_header(path, {
        "core": {"dtype": "BF16", "shape": [core_elements],
                 "data_offsets": [0, core_bytes]},
        "island": {"dtype": "F32", "shape": [1],
                   "data_offsets": [core_bytes, core_bytes + 4]},
    })
    assert sniff_fsdp_launch_quant(path) == expected


@pytest.mark.parametrize("suffix", [".ckpt", ".bin"])
def test_fsdp_launch_quant_defers_legacy_formats(tmp_path: Path, suffix):
    path = tmp_path / f"model{suffix}"
    path.write_bytes(b"not a safetensors container")
    assert sniff_fsdp_launch_quant(path) is None


def test_fsdp_launch_quant_malformed_safetensors_is_unknown(tmp_path: Path):
    path = tmp_path / "malformed.safetensors"
    path.write_bytes(b"not a safetensors container")
    assert sniff_fsdp_launch_quant(path) == "unknown"


def test_fsdp_lora_admission_property_admits_a_plain_fp8_dtype_checkpoint(
    tmp_path: Path,
):
    """Qwen-Image's literal float8_e4m3fn weights carry no scaled_fp8 marker
    and render with a LoRA stack under FSDP (2026-09 stable-source sweep)."""
    path = tmp_path / "qwen_fp8.safetensors"
    _write_dtype_checkpoint(path, ["F8_E4M3"])
    assert fsdp_lora_admission_property(path) is None


def test_fsdp_lora_admission_property_admits_a_uniform_bf16_checkpoint(
    tmp_path: Path,
):
    path = tmp_path / "bf16.safetensors"
    _write_dtype_checkpoint(path, ["BF16", "BF16"])
    assert fsdp_lora_admission_property(path) is None


def test_fsdp_lora_admission_property_refuses_a_scaled_fp8_marker(
    tmp_path: Path,
):
    """The scaled_fp8 sentinel tensor is what tells comfy to build the
    comfy-kitchen QuantizedTensor wrapper; a plain F8 dtype header without it
    stays admitted (the case above)."""
    path = tmp_path / "chroma_scaled_fp8.safetensors"
    _write_safetensors_header(path, {
        "scaled_fp8": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]},
        "blocks.0.w": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [1, 2]},
    })
    assert fsdp_lora_admission_property(path) == "quantized_shards"


def test_fsdp_lora_admission_property_refuses_int8(tmp_path: Path):
    path = tmp_path / "int8.safetensors"
    _write_dtype_checkpoint(path, ["I8"])
    assert fsdp_lora_admission_property(path) == "quantized_shards"


def test_fsdp_lora_admission_property_does_not_cover_fp32_islands(
    tmp_path: Path,
):
    """1 fp32 element beside 1000 bf16 ones is inside the base launch's 5%
    ceiling (test_fsdp_launch_quant_admits_fp32_islands_inside_the_ceiling
    above reads it as an admitted bf16 core). This header-only property
    check answers None for it whether the island stays fp32 through Comfy's
    load or casts to bf16; fsdp_lora_admission_property's docstring says why
    the header cannot tell. Krea2 RAW bf16 and ChromaRadiance carry the same
    header shape and differ on that cast (docs/TROUBLESHOOTING.md #17)."""
    path = tmp_path / "krea2_raw.safetensors"
    core_elements = 1000
    core_bytes = 2 * core_elements
    _write_safetensors_header(path, {
        "core": {"dtype": "BF16", "shape": [core_elements],
                 "data_offsets": [0, core_bytes]},
        "island": {"dtype": "F32", "shape": [1],
                   "data_offsets": [core_bytes, core_bytes + 4]},
    })
    assert sniff_fsdp_launch_quant(path) == "bf16"  # base launch: admitted
    assert fsdp_lora_admission_property(path) is None


def test_fsdp_lora_admission_property_none_for_an_fp32_share_over_the_ceiling_too(
    tmp_path: Path,
):
    """Over the ceiling the base launch already refuses (quant=fp32); the
    LoRA-specific property check answers None here too, since the coarser
    base-launch refusal (validate_fsdp_launch_quant) always answers first
    for a non-bf16-kind checkpoint."""
    path = tmp_path / "mostly_fp32.safetensors"
    _write_safetensors_header(path, {
        "core": {"dtype": "BF16", "shape": [30], "data_offsets": [0, 60]},
        "island": {"dtype": "F32", "shape": [1], "data_offsets": [60, 64]},
    })
    assert sniff_fsdp_launch_quant(path) == "fp32"  # base launch: refused
    assert fsdp_lora_admission_property(path) is None


@pytest.mark.parametrize("marker_dtype", ["F8_E4M3", "I8"])
def test_fsdp_lora_admission_property_refuses_a_comfy_quant_descriptor(
    tmp_path: Path, marker_dtype,
):
    """comfy-kitchen's per-tensor scaled fp8mixed release carries a
    `*.comfy_quant` U8 descriptor beside the quantized weight, with no
    scaled_fp8 marker and no __metadata__ (confirmed against the on-disk
    Chroma1-HD-fp8mixed.safetensors header)."""
    path = tmp_path / "fp8mixed.safetensors"
    _write_safetensors_header(path, {
        "blocks.0.w.weight": {"dtype": marker_dtype, "shape": [1], "data_offsets": [0, 1]},
        "blocks.0.w.comfy_quant": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
    })
    assert fsdp_lora_admission_property(path) == "quantized_shards"


def test_fsdp_lora_admission_property_refuses_quantization_metadata(
    tmp_path: Path,
):
    """flux2_dev_fp8mixed and krea2_raw_fp8_scaled carry an
    `_quantization_metadata` __metadata__ key naming the per-layer format,
    with no scaled_fp8 tensor marker (headers read 2026-09-28)."""
    path = tmp_path / "flux2_fp8mixed.safetensors"
    _write_safetensors_header(path, {
        "__metadata__": {"_quantization_metadata": '{"format_version": "1.0"}'},
        "blocks.0.w.weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]},
    })
    assert fsdp_lora_admission_property(path) == "quantized_shards"


def test_fsdp_lora_admission_property_refuses_a_weight_scale_beside_f8(
    tmp_path: Path,
):
    """A *.weight_scale sidecar beside an F8-dtype weight is the marker even
    with no scaled_fp8 tensor and no __metadata__ (confirmed against the
    on-disk Chroma1-HD-fp8mixed.safetensors header)."""
    path = tmp_path / "scaled.safetensors"
    _write_safetensors_header(path, {
        "blocks.0.w.weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]},
        "blocks.0.w.weight_scale": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]},
    })
    assert fsdp_lora_admission_property(path) == "quantized_shards"


def test_fsdp_lora_admission_property_admits_a_weight_scale_beside_bf16(
    tmp_path: Path,
):
    """A tensor named '*.weight_scale' beside a full-precision weight
    is not evidence of quantization; only a scale beside an F8/I8 weight is."""
    path = tmp_path / "unrelated_scale.safetensors"
    _write_safetensors_header(path, {
        "norm.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "norm.weight_scale": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
    })
    assert fsdp_lora_admission_property(path) is None


# Hardware ground truth: every LoRA+FSDP cell in the 2026-09 validation
# campaign, read off the real checkpoint headers under
# ~/ComfyUI/models/diffusion_models. Skipped file by file when a name is
# absent, so CI without the model directory stays green; every name present
# must match. krea2_raw_bf16 and chroma-radiance-x0 carry fp32 islands, which
# this header check does not classify, so it admits both (the fp32-islands test
# above). krea2_raw_bf16 is the one campaign row that still needs the in-bake
# backstop (actor/fsdp_lora.py) rather than this header check.
# chroma-radiance-x0 is admitted from code, not proven by a LoRA cell of its
# own: the 2026-08-26 base FSDP census found its 2 fp32 island parameters
# replicated at fp32 (docs/VALIDATION.md, Radiance evidence history;
# docs/TROUBLESHOOTING.md #17).
_ORACLE_DIR = Path("~/ComfyUI/models/diffusion_models").expanduser()
_ORACLE_REFUSE = (
    "Chroma1-HD-fp8_scaled_defaultloader_hybrid_large_rev2.safetensors",
    "Chroma1-HD-fp8mixed.safetensors",
    "Chroma1-HD-int8_convrot.safetensors",
    "Krea2_Turbo_convrot_int8mixed.safetensors",
    "flux2_dev_fp8mixed.safetensors",
    "krea2_raw_convrot_int8.safetensors",
    "krea2_raw_fp8_scaled.safetensors",
)
_ORACLE_ADMIT = (
    "Chroma1-HD.safetensors",
    "flux2-dev.safetensors",
    "kandinsky5lite_t2i.safetensors",
    "qwen_image_2512_fp8_e4m3fn.safetensors",
    "z_image_turbo_bf16.safetensors",
    # fp32-island files (the comment above).
    "krea2_raw_bf16.safetensors",
    "chroma-radiance-x0.safetensors",
)


@pytest.mark.parametrize("name", _ORACLE_REFUSE)
def test_fsdp_lora_admission_oracle_refuses_the_real_checkpoint(name):
    path = _ORACLE_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} not present on this box")
    assert fsdp_lora_admission_property(path) == "quantized_shards"


@pytest.mark.parametrize("name", _ORACLE_ADMIT)
def test_fsdp_lora_admission_oracle_admits_the_real_checkpoint(name):
    path = _ORACLE_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} not present on this box")
    assert fsdp_lora_admission_property(path) is None


@pytest.mark.parametrize("suffix", [".ckpt", ".bin"])
def test_fsdp_lora_admission_property_fails_open_for_legacy_formats(
    tmp_path: Path, suffix,
):
    path = tmp_path / f"model{suffix}"
    path.write_bytes(b"not a safetensors container")
    assert fsdp_lora_admission_property(path) is None


def test_fsdp_lora_admission_property_fails_open_for_malformed_safetensors(
    tmp_path: Path,
):
    path = tmp_path / "malformed.safetensors"
    path.write_bytes(b"not a safetensors container")
    assert fsdp_lora_admission_property(path) is None


def test_truncated_file_raises_typed(tmp_path: Path):
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(CheckpointSniffError):
        sniff_checkpoint(path)


@pytest.mark.parametrize("header", [[], "not-an-object", 3, None])
def test_non_object_json_header_raises_typed(tmp_path: Path, header):
    path = tmp_path / "malformed.safetensors"
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(CheckpointSniffError, match="must be an object"):
        sniff_checkpoint(path)


@pytest.mark.parametrize("metadata", [[], "", 0, False])
def test_falsey_non_object_metadata_raises_typed(tmp_path: Path, metadata):
    path = tmp_path / "bad_metadata.safetensors"
    _write_safetensors_header(path, {"__metadata__": metadata})
    with pytest.raises(CheckpointSniffError, match="__metadata__ must be an object"):
        sniff_checkpoint(path)


def test_malformed_tensor_descriptor_raises_typed(tmp_path: Path):
    path = tmp_path / "malformed.safetensors"
    raw = json.dumps({"weight": ["not", "a", "descriptor"]}).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(CheckpointSniffError, match="descriptor"):
        sniff_checkpoint(path)


def test_declared_header_length_must_be_fully_present(tmp_path: Path):
    path = tmp_path / "truncated.safetensors"
    raw = b'{"weight":'
    path.write_bytes(struct.pack("<Q", len(raw) + 5) + raw)
    with pytest.raises(CheckpointSniffError, match="truncated"):
        sniff_checkpoint(path)


def test_tensor_payload_must_be_fully_present(tmp_path: Path):
    path = tmp_path / "truncated_data.safetensors"
    descriptor = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    raw = json.dumps(descriptor).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    with pytest.raises(CheckpointSniffError, match="exceeds file size"):
        sniff_checkpoint(path)


def test_tensor_shape_must_match_byte_range(tmp_path: Path):
    path = tmp_path / "shape_mismatch.safetensors"
    descriptor = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}
    _write_safetensors_header(path, descriptor)
    with pytest.raises(CheckpointSniffError, match="dtype/shape require 8"):
        sniff_checkpoint(path)


def test_tensor_ranges_cannot_overlap(tmp_path: Path):
    path = tmp_path / "bad_ranges.safetensors"
    _write_safetensors_header(path, {
        "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
    })
    with pytest.raises(CheckpointSniffError, match="overlaps"):
        sniff_checkpoint(path)


def test_alignment_padding_holes_are_a_legal_layout(tmp_path: Path):
    """Some producers pad between tensors; each descriptor is still bounds-
    and size-checked, and overlap remains a hard error."""
    path = tmp_path / "padded_ranges.safetensors"
    _write_safetensors_header(path, {
        "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "b": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]},
    })
    sniff_checkpoint(path)  # must not raise


def test_quant_from_scaled_fp8_key():
    keys = ["scaled_fp8", "blocks.0.w"]
    assert detect_quant_from_header(keys, {"__tensors__": {}}) == "fp8"


def test_quant_from_dtype():
    header = {"__tensors__": {"w": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}}}
    assert detect_quant_from_header(["w"], header) == "fp8"
    header_bf16 = {"__tensors__": {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}}
    assert detect_quant_from_header(["w"], header_bf16) == "bf16"
    header_fp16 = {"__tensors__": {"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}}
    assert detect_quant_from_header(["w"], header_fp16) == "fp16"
    header_f4 = {"__tensors__": {"w": {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]}}}
    assert detect_quant_from_header(["w"], header_f4) == "fp8"


@given(extra_dtypes=st.sets(st.sampled_from(("BF16", "F32", "U8")), max_size=3))
def test_any_fp16_tensor_fails_closed_to_fp16(extra_dtypes):
    dtypes = ["F16", *sorted(extra_dtypes)]
    tensors = {
        f"w{index}": {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}
        for index, dtype in enumerate(dtypes)
    }
    assert detect_quant_from_header(list(tensors), {"__tensors__": tensors}) == "fp16"


@pytest.mark.parametrize(("quant_dtype", "expected"), [
    ("F8_E4M3", "fp8"),
    ("F4", "fp8"),
    ("I8", "int8"),
])
def test_quantized_marker_takes_precedence_over_fp16(quant_dtype, expected):
    tensors = {
        "full.weight": {"dtype": "F16"},
        "quant.weight": {"dtype": quant_dtype},
    }
    assert detect_quant_from_header(list(tensors), {"__tensors__": tensors}) == expected


@pytest.mark.parametrize("dtype", ["BF16", "F32"])
def test_non_fp16_full_precision_keeps_bf16_compatibility_default(dtype):
    tensors = {"w": {"dtype": dtype}}
    assert detect_quant_from_header(["w"], {"__tensors__": tensors}) == "bf16"


def test_sniff_end_to_end(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    _write_safetensors_header(path, {
        "txtfusion.projector.weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]},
        "blocks.0.attn.wq.weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [1, 2]},
        "__metadata__": {"format": "pt"},
    })
    family, quant = sniff_checkpoint(path)
    assert family == "krea2"
    assert quant == "fp8"


def test_hunyuan_refiner_sniff_end_to_end(tmp_path: Path):
    path = tmp_path / "hunyuan_image_refiner.safetensors"
    tensors = {
        key: {
            "dtype": "F8_E4M3",
            "shape": [1],
            "data_offsets": [index, index + 1],
        }
        for index, key in enumerate(HUNYUAN_REFINER_KEYS)
    }
    tensors["__metadata__"] = {"format": "pt"}
    _write_safetensors_header(path, tensors)

    assert sniff_checkpoint(path) == ("hunyuan", "fp8")


def test_sniff_fp16_end_to_end(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    _write_safetensors_header(path, {
        "txtfusion.projector.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        "blocks.0.attn.wq.weight": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]},
        "__metadata__": {"format": "pt"},
    })

    assert sniff_checkpoint(path) == ("krea2", "fp16")


def _write_ltx_checkpoint(path: Path, dtypes: list[str]) -> None:
    """An LTX-keyed header whose tensors carry the given dtypes, in order."""
    widths = {"BF16": 2, "F32": 4}
    offset = 0
    tensors: dict[str, dict] = {}
    for key, dtype in zip(FAMILY_KEYS["ltx"], dtypes, strict=True):
        width = widths[dtype]
        tensors[key] = {"dtype": dtype, "shape": [1],
                        "data_offsets": [offset, offset + width]}
        offset += width
    tensors["__metadata__"] = {"format": "pt"}
    _write_safetensors_header(path, tensors)


def test_topology_sniff_names_an_all_fp32_checkpoint_apart(tmp_path: Path):
    # The auto table must not route an all-FP32 file by a BF16 row. The loose
    # sniff still answers its compatibility default, which every family-only
    # caller reads.
    path = tmp_path / "all_fp32.safetensors"
    _write_ltx_checkpoint(path, ["F32", "F32", "F32"])

    assert sniff_checkpoint(path) == ("ltx", "bf16")
    assert sniff_checkpoint_for_topology(path) == ("ltx", "fp32")


def test_topology_sniff_keeps_bf16_for_a_bf16_core_with_fp32_auxiliaries(
    tmp_path: Path,
):
    # The shape of the public LTX 2.5 BF16 export: a BF16 core carrying FP32
    # AdaLN tables. It is a BF16 export and must keep routing as one.
    path = tmp_path / "bf16_core.safetensors"
    _write_ltx_checkpoint(path, ["F32", "BF16", "BF16"])

    assert sniff_checkpoint(path) == ("ltx", "bf16")
    assert sniff_checkpoint_for_topology(path) == ("ltx", "bf16")


@pytest.mark.parametrize("dtypes", [["BF16", "BF16", "BF16"]])
def test_topology_sniff_agrees_with_the_loose_sniff_on_plain_bf16(
    tmp_path: Path, dtypes,
):
    path = tmp_path / "plain_bf16.safetensors"
    _write_ltx_checkpoint(path, dtypes)

    assert sniff_checkpoint(path) == sniff_checkpoint_for_topology(path)
    assert sniff_checkpoint_for_topology(path) == ("ltx", "bf16")


def _row_label(index: int) -> str:
    """A stable name for one signature row, including a family's twins."""
    family, _needles = _SIGNATURES[index]
    twins = [i for i, (f, _n) in enumerate(_SIGNATURES) if f == family]
    return family if len(twins) == 1 else f"{family}[{twins.index(index)}]"


# Every ordered pair of rows sharing a key path, earlier row first, with the
# needles each row holds that the other does not. The table is written as
# specific-to-general chains, so these overlaps are by design. Two things must
# stay true: the earlier row of every pair carries a needle the later one
# lacks, and every overlap is declared here.
_SHADOW_PAIRS: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {
    ("hunyuan[0]", "hunyuan[1]"): (("cond_type_embedding.",), ("byt5_in.",)),
    ("hunyuan[0]", "hunyuan[2]"): (("cond_type_embedding.",), ("time_r_in.",)),
    ("hunyuan[1]", "hunyuan[2]"): (("byt5_in.",), ("time_r_in.",)),
    ("flux", "longcat"): (("vector_in.",), ()),
    ("qwen_image21", "lens"): (("img_in.", "modulation.1."),
                              ("attn.norm_added_q.", "img_mlp.w1.")),
    ("qwen_image21", "qwen_image"): (("modulation.1.",), ("txt_norm.",)),
    ("qwen_image21", "ltx"): (("img_in.", "modulation.1."),
                             ("adaln_single.", "patchify_proj.")),
    ("lens", "qwen_image"): (("attn.norm_added_q.", "img_mlp.w1."),
                             ("img_in.", "txt_norm.")),
    ("lens", "ltx"): (("attn.norm_added_q.", "img_mlp.w1."),
                      ("adaln_single.", "patchify_proj.")),
    ("qwen_image", "ltx"): (("img_in.", "txt_norm."),
                            ("adaln_single.", "patchify_proj.")),
    ("cogvideo[0]", "cogvideo[1]"): (("time_embedding_linear_1.",),
                                     ("time_embedding.linear_1.",)),
    ("wan_scail", "wan_dancer"): (("patch_embedding_pose.",),
                                  ("patch_embedding_global.",)),
    ("wan_scail", "wan"): (("patch_embedding_pose.",), ("self_attn.",)),
    ("wan_dancer", "wan"): (("patch_embedding_global.",), ("self_attn.",)),
}


def test_every_overlapping_signature_pair_is_declared():
    """Adding or editing a needle that creates a new overlap fails here.

    One pair is a strict subset: longcat's three paths are flux's four minus
    `vector_in.`, so every healthy longcat file has matched three of flux's
    four by construction. That is why no runtime partial-match report can
    separate a healthy longcat from a damaged flux, and why this ledger is a
    static test rather than a message a render prints.
    """
    computed = {}
    for earlier in range(len(_SIGNATURES)):
        for later in range(earlier + 1, len(_SIGNATURES)):
            first = set(_SIGNATURES[earlier][1])
            second = set(_SIGNATURES[later][1])
            if first & second:
                computed[(_row_label(earlier), _row_label(later))] = (
                    tuple(sorted(first - second)), tuple(sorted(second - first)))
    assert computed == _SHADOW_PAIRS


def test_no_shadowed_row_is_fully_covered_by_a_later_one():
    """If a later row held every needle of an earlier one, the later row would
    be dead code: the earlier row matches every header it would, and first."""
    for (earlier, _later), (only_earlier, _only_later) in _SHADOW_PAIRS.items():
        assert only_earlier, f"{earlier} carries nothing the later row lacks"


@pytest.mark.parametrize(("keys", "expected"), [
    (["weird.wrap.txtfusion.projector.weight"], "krea2"),
    (["mystery.text_embedding.0.weight",
      "mystery.blocks.0.self_attn.q.weight"], "wan"),
    (["wrap.cap_embedder.weight", "wrap.x_pad_token"], "zimage"),
    (["w.adaln_single.linear.weight", "w.transformer_blocks.0.attn1.to_q.weight",
      "w.patchify_proj.weight"], "ltx"),
])
def test_a_novel_top_level_prefix_still_detects_every_shape(keys, expected):
    """`_has_segment` matches at dotted boundaries wherever they fall, so an
    export under an unlisted wrapper still detects. `_strip_prefix` does not
    decide these matches, so `unknown` means "no signature matched", never
    "your prefix is not on the list". Anchoring needles at position 0 would
    break all four of these.
    """
    assert detect_family_from_keys(keys) == expected


def test_a_pixart_layout_names_ltx_as_the_closest_signature():
    keys = ["adaln_single.linear.weight", "transformer_blocks.0.attn1.to_q.weight",
            "pos_embed.proj.weight"]
    assert detect_family_from_keys(keys) == "unknown"
    best = nearest_signatures(keys)[0]
    assert (best.family, best.present, best.total) == ("ltx", 2, 3)
    assert best.missing == ("patchify_proj.",)


def test_a_row_that_matched_outright_is_never_a_near_miss():
    for near in nearest_signatures(FAMILY_KEYS["ltx"]):
        assert near.missing
        assert near.family != "ltx"


def test_a_header_matching_nothing_at_all_reports_no_near_miss():
    assert nearest_signatures(["some.random.weight"]) == ()


def test_the_cached_near_miss_accessor_agrees_with_the_pure_helper(tmp_path: Path):
    path = tmp_path / "pixart.safetensors"
    _write_safetensors_header(path, {
        "adaln_single.linear.weight": {"dtype": "BF16", "shape": [1],
                                       "data_offsets": [0, 2]},
        "transformer_blocks.0.attn1.to_q.weight": {"dtype": "BF16", "shape": [1],
                                                   "data_offsets": [2, 4]},
        "pos_embed.proj.weight": {"dtype": "BF16", "shape": [1],
                                  "data_offsets": [4, 6]},
        "__metadata__": {"format": "pt"},
    })
    cached = sniff_nearest_signatures(path)
    assert cached[0].family == "ltx"
    assert cached == sniff_nearest_signatures(path)


def _write_metadata_checkpoint(path: Path, metadata: dict) -> None:
    _write_safetensors_header(path, {
        "txtfusion.projector.weight": {"dtype": "BF16", "shape": [1],
                                       "data_offsets": [0, 2]},
        "__metadata__": metadata,
    })


def test_metadata_keys_come_back_with_the_memoized_sniff(tmp_path: Path):
    path = tmp_path / "with_config.safetensors"
    _write_metadata_checkpoint(path, {"config": "{}", "license": "apache-2.0"})
    assert sniff_metadata_keys(path) == ("config", "license")
    assert sniff_checkpoint(path) == ("krea2", "bf16")


def test_a_bare_header_reports_no_metadata_keys(tmp_path: Path):
    path = tmp_path / "bare.safetensors"
    _write_safetensors_header(path, {
        "txtfusion.projector.weight": {"dtype": "BF16", "shape": [1],
                                       "data_offsets": [0, 2]},
    })
    assert sniff_metadata_keys(path) == ()


def test_the_stat_failure_path_answers_the_same_four_values(tmp_path: Path,
                                                            monkeypatch):
    """The cached read and the stat-failure read must not answer different
    shapes; a 3-tuple on one side is a ValueError at unpack, on a race."""
    path = tmp_path / "raced.safetensors"
    _write_metadata_checkpoint(path, {"config": "{}"})
    cached = _sniff(path)
    assert len(cached) == 4

    real_stat = Path.stat

    def refuse_stat(self, *args, **kwargs):
        if str(self) == str(path):
            raise OSError("stat raced the read")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", refuse_stat)
    assert _sniff(path) == cached
    assert _read_sniff(path) == cached


# Z-Image ships three surfaces on one backbone, and all three detect as
# `zimage`. The L2P pixel-space checkpoint carries the local_decoder weights
# and the DCT PixelSpace one the dec_net head. ComfyUI reads DCT as pixel space
# and, as of 2026-09-02, L2P as latent Z-Image.
ZIMAGE_KEYS = [
    "cap_embedder.0.weight",
    "x_pad_token",
    "layers.0.attention.qkv.weight",
]
ZIMAGE_L2P_KEYS = [
    *ZIMAGE_KEYS,
    "all_x_embedder.16-1.weight",
    "local_decoder.blocks.0.weight",
]
ZIMAGE_DCT_KEYS = [*ZIMAGE_KEYS, "dec_net.blocks.0.weight"]


def _write_key_checkpoint(path: Path, keys: list[str]) -> Path:
    tensors = {
        key: {"dtype": "BF16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]}
        for index, key in enumerate(keys)
    }
    _write_safetensors_header(path, tensors)
    return path


def test_an_l2p_header_is_named_apart_from_plain_zimage():
    assert detect_family_from_keys(ZIMAGE_L2P_KEYS) == "zimage"
    assert is_zimage_l2p_from_keys(ZIMAGE_L2P_KEYS) is True


def test_a_plain_latent_zimage_header_is_not_l2p():
    assert detect_family_from_keys(ZIMAGE_KEYS) == "zimage"
    assert is_zimage_l2p_from_keys(ZIMAGE_KEYS) is False


def test_a_dct_pixelspace_header_is_not_l2p():
    """DCT PixelSpace is the near neighbour the L2P reading must tell apart."""
    assert detect_family_from_keys(ZIMAGE_DCT_KEYS) == "zimage"
    assert is_zimage_l2p_from_keys(ZIMAGE_DCT_KEYS) is False


def test_a_header_carrying_both_decoders_is_not_l2p():
    """The safe direction: an unrecognized mixture loads rather than refuses."""
    assert is_zimage_l2p_from_keys([*ZIMAGE_L2P_KEYS, "dec_net.blocks.0.weight"]) is False


def test_a_non_zimage_header_carrying_local_decoder_is_not_l2p():
    keys = [*FAMILY_KEYS["krea2"], "local_decoder.blocks.0.weight"]
    assert detect_family_from_keys(keys) == "krea2"
    assert is_zimage_l2p_from_keys(keys) is False


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model.", "diffusion_model."])
def test_the_l2p_reading_survives_the_comfy_wrapper_prefixes(prefix):
    assert is_zimage_l2p_from_keys([prefix + k for k in ZIMAGE_L2P_KEYS]) is True


def test_the_l2p_read_does_not_change_what_sniff_checkpoint_returns(tmp_path: Path):
    """A new reading of the same header, not a new family word.

    Nothing downstream may see an `l2p` family: topology selection, the adapter
    registry and every memo keyed by family still read `zimage`.
    """
    path = _write_key_checkpoint(tmp_path / "l2p.safetensors", ZIMAGE_L2P_KEYS)
    assert sniff_checkpoint(path) == ("zimage", "bf16")
    assert sniff_checkpoint_for_topology(path) == ("zimage", "bf16")
    assert sniff_zimage_l2p(path) is True

    plain = _write_key_checkpoint(tmp_path / "latent.safetensors", ZIMAGE_KEYS)
    assert sniff_checkpoint(plain) == ("zimage", "bf16")
    assert sniff_zimage_l2p(plain) is False


def test_an_unreadable_or_non_safetensors_path_is_not_l2p(tmp_path: Path):
    """Total, and quiet: a read that cannot answer never invents a refusal."""
    assert sniff_zimage_l2p(tmp_path / "absent.safetensors") is False
    assert sniff_zimage_l2p(tmp_path) is False
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(b"\x00\x01")
    assert sniff_zimage_l2p(truncated) is False
    not_safetensors = tmp_path / "model.ckpt"
    not_safetensors.write_bytes(b"pickle")
    assert sniff_zimage_l2p(not_safetensors) is False


def test_the_acceptance_log_line_names_a_nested_decoder_spelling(
        tmp_path: Path, caplog):
    """The INFO line is the evidence the hardware acceptance reads.

    The predicate matches a decoder path at any dotted boundary, so the line
    that reports which spelling matched must use the same rule. A plain
    ``startswith`` filter answers the refusal correctly and then logs an empty
    list, which reads as a gate that fired on nothing.
    """
    nested = "model.diffusion_model.zimage.local_decoder.blocks.0.weight"
    keys = [*ZIMAGE_KEYS, nested]
    assert is_zimage_l2p_from_keys(keys) is True
    path = _write_key_checkpoint(tmp_path / "nested_l2p.safetensors", keys)
    with caplog.at_level("INFO", logger="dgx_monarch.adapters.detect"):
        assert sniff_zimage_l2p(path) is True
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "zimage.local_decoder.blocks.0.weight" in logged
    assert "dec_net" in logged


# Both real layouts as the lens_bf16 and mage_flow_bf16 headers list them on
# 2026-10-05: every top-level tensor plus block 0, names and shapes only (the
# other 47 Lens and 11 Mage blocks repeat block 0). Mage carries every path
# of the qwen_image row; Lens carries them too, with its own block keys. Lens
# also carries Mage's 128-wide `img_in` input and the `txt_norm.` and
# `proj_out.` paths, so a Mage test on the `img_in` input width, run ahead of
# the key table, reads the real Lens file as mage_flow (docs/VALIDATION.md,
# "Lens and Mage Flow header sniff").
LENS_BF16_LAYOUT = {
    "img_in.weight": [1536, 128], "img_in.bias": [1536],
    "txt_in.weight": [1536, 11520], "txt_in.bias": [1536],
    "txt_norm.0.weight": [2880], "txt_norm.1.weight": [2880],
    "txt_norm.2.weight": [2880], "txt_norm.3.weight": [2880],
    "time_text_embed.timestep_embedder.linear_1.weight": [1536, 256],
    "time_text_embed.timestep_embedder.linear_1.bias": [1536],
    "time_text_embed.timestep_embedder.linear_2.weight": [1536, 1536],
    "time_text_embed.timestep_embedder.linear_2.bias": [1536],
    "norm_out.linear.weight": [3072, 1536], "norm_out.linear.bias": [3072],
    "proj_out.weight": [128, 1536], "proj_out.bias": [128],
    "transformer_blocks.0.attn.img_qkv.weight": [4608, 1536],
    "transformer_blocks.0.attn.img_qkv.bias": [4608],
    "transformer_blocks.0.attn.txt_qkv.weight": [4608, 1536],
    "transformer_blocks.0.attn.txt_qkv.bias": [4608],
    "transformer_blocks.0.attn.norm_q.weight": [64],
    "transformer_blocks.0.attn.norm_k.weight": [64],
    "transformer_blocks.0.attn.norm_added_q.weight": [64],
    "transformer_blocks.0.attn.norm_added_k.weight": [64],
    "transformer_blocks.0.attn.to_out.0.weight": [1536, 1536],
    "transformer_blocks.0.attn.to_out.0.bias": [1536],
    "transformer_blocks.0.attn.to_add_out.weight": [1536, 1536],
    "transformer_blocks.0.attn.to_add_out.bias": [1536],
    "transformer_blocks.0.img_mlp.w1.weight": [4096, 1536],
    "transformer_blocks.0.img_mlp.w2.weight": [1536, 4096],
    "transformer_blocks.0.img_mlp.w3.weight": [4096, 1536],
    "transformer_blocks.0.txt_mlp.w1.weight": [4096, 1536],
    "transformer_blocks.0.txt_mlp.w2.weight": [1536, 4096],
    "transformer_blocks.0.txt_mlp.w3.weight": [4096, 1536],
    "transformer_blocks.0.img_mod.1.weight": [9216, 1536],
    "transformer_blocks.0.img_mod.1.bias": [9216],
    "transformer_blocks.0.txt_mod.1.weight": [9216, 1536],
    "transformer_blocks.0.txt_mod.1.bias": [9216],
    "transformer_blocks.0.img_norm1.weight": [1536],
    "transformer_blocks.0.img_norm2.weight": [1536],
    "transformer_blocks.0.txt_norm1.weight": [1536],
    "transformer_blocks.0.txt_norm2.weight": [1536],
}

MAGE_FLOW_BF16_LAYOUT = {
    "img_in.weight": [3072, 128], "img_in.bias": [3072],
    "txt_in.weight": [3072, 2560], "txt_in.bias": [3072],
    "txt_norm.weight": [2560],
    "time_text_embed.timestep_embedder.linear_1.weight": [3072, 256],
    "time_text_embed.timestep_embedder.linear_1.bias": [3072],
    "time_text_embed.timestep_embedder.linear_2.weight": [3072, 3072],
    "time_text_embed.timestep_embedder.linear_2.bias": [3072],
    "norm_out.linear.weight": [6144, 3072], "norm_out.linear.bias": [6144],
    "proj_out.weight": [128, 3072], "proj_out.bias": [128],
    **{f"transformer_blocks.0.attn.{name}.{leaf}": shape
       for name in ("to_q", "to_k", "to_v", "to_out.0",
                    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")
       for leaf, shape in (("weight", [3072, 3072]), ("bias", [3072]))},
    "transformer_blocks.0.attn.norm_q.weight": [128],
    "transformer_blocks.0.attn.norm_k.weight": [128],
    "transformer_blocks.0.attn.norm_added_q.weight": [128],
    "transformer_blocks.0.attn.norm_added_k.weight": [128],
    **{f"transformer_blocks.0.{stream}_mlp.{name}.{leaf}": shape
       for stream in ("img", "txt")
       for name, leaf, shape in (("net.0.proj", "weight", [12288, 3072]),
                                 ("net.0.proj", "bias", [12288]),
                                 ("net.2", "weight", [3072, 12288]),
                                 ("net.2", "bias", [3072]))},
    "transformer_blocks.0.img_mod.1.weight": [18432, 3072],
    "transformer_blocks.0.img_mod.1.bias": [18432],
    "transformer_blocks.0.txt_mod.1.weight": [18432, 3072],
    "transformer_blocks.0.txt_mod.1.bias": [18432],
}


def _bf16_tensors(layout: dict[str, list[int]]) -> dict[str, dict]:
    return {key: {"dtype": "BF16", "shape": shape} for key, shape in layout.items()}


def _int8_convrot_tensors(layout: dict[str, list[int]]) -> dict[str, dict]:
    """The mage_flow_turbo_int8_convrot header: every 2-D block weight stored
    I8 beside a per-row F32 scale and a U8 comfy_quant descriptor (14 weights
    per block, 336 extra keys in the real file); everything else stays BF16."""
    tensors = {}
    for key, shape in layout.items():
        quantized = (key.startswith("transformer_blocks.") and key.endswith(".weight")
                     and len(shape) == 2)
        tensors[key] = {"dtype": "I8" if quantized else "BF16", "shape": shape}
        if quantized:
            module = key[: -len(".weight")]
            tensors[f"{module}.weight_scale"] = {"dtype": "F32", "shape": [shape[0], 1]}
            tensors[f"{module}.comfy_quant"] = {"dtype": "U8", "shape": [72]}
    return tensors


def _family(tensors: dict[str, dict]) -> str:
    return detect_family_from_header(list(tensors), {"__tensors__": tensors})


def test_the_real_lens_layout_reads_lens():
    assert _family(_bf16_tensors(LENS_BF16_LAYOUT)) == "lens"


def test_the_real_mage_flow_layout_reads_mage_flow():
    """By key alone Mage is the qwen_image row; its widths name it apart."""
    assert detect_family_from_keys(list(MAGE_FLOW_BF16_LAYOUT)) == "qwen_image"
    assert _family(_bf16_tensors(MAGE_FLOW_BF16_LAYOUT)) == "mage_flow"


def test_the_real_int8_mage_flow_layout_reads_mage_flow_int8():
    tensors = _int8_convrot_tensors(MAGE_FLOW_BF16_LAYOUT)
    header = {"__tensors__": tensors}
    assert _family(tensors) == "mage_flow"
    assert detect_quant_from_header(list(tensors), header) == "int8"


def test_lens_block_keys_win_over_mage_widths():
    """The order ComfyUI keeps: Lens's block keys are tested before Mage's
    widths. The real Lens file fails the width test on its four `txt_norm.N`
    layers; a single-layer Lens export carrying Mage's 2560-wide `txt_norm`
    and 128-row `proj_out` must still read lens."""
    layout = {k: v for k, v in LENS_BF16_LAYOUT.items() if not k.startswith("txt_norm.")}
    layout["txt_norm.weight"] = [2560]
    assert layout["proj_out.weight"][0] == 128
    assert _family(_bf16_tensors(layout)) == "lens"


@pytest.mark.parametrize(("txt_norm", "proj_out"), [
    ([3584], [64, 3072]),   # Qwen Image's real widths (qwen_image_2512 header)
    ([2560], [64, 3072]),
    ([3584], [128, 3072]),
], ids=["qwen_image_widths", "mage_txt_norm_only", "mage_proj_out_only"])
def test_anything_short_of_both_mage_widths_stays_qwen_image(txt_norm, proj_out):
    layout = {**MAGE_FLOW_BF16_LAYOUT, "txt_norm.weight": txt_norm,
              "proj_out.weight": proj_out}
    assert _family(_bf16_tensors(layout)) == "qwen_image"


def test_mage_flow_widths_read_first_dimensions_only():
    """A 4-bit export halves a linear's second dimension, so a width test on
    a second dimension (`img_in` input channels) misses a 4-bit Mage file."""
    layout = {**MAGE_FLOW_BF16_LAYOUT, "img_in.weight": [3072, 64],
              "proj_out.weight": [128, 1536]}
    assert _family(_bf16_tensors(layout)) == "mage_flow"


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
def test_lens_and_mage_flow_survive_the_comfy_wrapper_prefixes(prefix):
    for layout, family in ((LENS_BF16_LAYOUT, "lens"), (MAGE_FLOW_BF16_LAYOUT, "mage_flow")):
        tensors = {prefix + k: v for k, v in _bf16_tensors(layout).items()}
        assert _family(tensors) == family


def _write_layout_checkpoint(path: Path, tensors: dict[str, dict]) -> Path:
    """On disk every trailing dimension shrinks to 1. The sniff reads first
    dimensions only, and the container check would otherwise want every real
    byte range written out."""
    widths = {"BF16": 2, "F32": 4, "I8": 1, "U8": 1}
    offset = 0
    header = {}
    for key, info in tensors.items():
        shape = [info["shape"][0], *([1] * (len(info["shape"]) - 1))]
        size = math.prod(shape) * widths[info["dtype"]]
        header[key] = {"dtype": info["dtype"], "shape": shape,
                       "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"format": "pt"}
    _write_safetensors_header(path, header)
    return path


@pytest.mark.parametrize(("name", "tensors", "expected"), [
    ("lens_bf16", _bf16_tensors(LENS_BF16_LAYOUT), ("lens", "bf16")),
    ("mage_flow_bf16", _bf16_tensors(MAGE_FLOW_BF16_LAYOUT), ("mage_flow", "bf16")),
    ("mage_flow_turbo_int8_convrot", _int8_convrot_tensors(MAGE_FLOW_BF16_LAYOUT),
     ("mage_flow", "int8")),
])
def test_both_real_layouts_sniff_end_to_end(tmp_path: Path, name, tensors, expected):
    path = _write_layout_checkpoint(tmp_path / f"{name}.safetensors", tensors)
    assert sniff_checkpoint(path) == expected
    assert sniff_checkpoint_for_topology(path) == expected


# The real headers, skipped file by file where absent (the _ORACLE_DIR rule
# above). The two Qwen files are the neighbours the Mage widths must not take.
_SNIFF_ORACLE = (
    ("lens_bf16.safetensors", ("lens", "bf16")),
    ("mage_flow_bf16.safetensors", ("mage_flow", "bf16")),
    ("mage_flow_edit_bf16.safetensors", ("mage_flow", "bf16")),
    ("mage_flow_turbo_int8_convrot.safetensors", ("mage_flow", "int8")),
    ("mage_flow_edit_turbo_int8_convrot.safetensors", ("mage_flow", "int8")),
    ("qwen_image_2512_fp8_e4m3fn.safetensors", ("qwen_image", "fp8")),
    ("qwen_image_2.1_bf16.safetensors", ("qwen_image21", "bf16")),
)


@pytest.mark.parametrize(("name", "expected"), _SNIFF_ORACLE)
def test_the_real_checkpoint_sniffs_its_own_family(name, expected):
    path = _ORACLE_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} not present on this box")
    assert sniff_checkpoint(path) == expected
