"""p-h3-image on the MiniMax H3 path: detection, the w4a8 kind, residency and the sweep.

ComfyUI builds the p-h3-image DiT as MiniMaxH3 from H3's keys (#17506), so it
reads as minimax_h3 here and runs on the H3 adapter; tests/test_p_h3_image_comfy_exactness.py
holds that forward to stock on the image graph. The headers below copy the
release's layout (PrunaAI/P-H3-Image-Comfy at 44a0fb73f299, docs/MODELS.md):
block 0 of each DiT file, the curve basis, and the token refiner's first linear.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.sweep import convert  # noqa: E402
from benchmark.sweep.matrix import split_quant  # noqa: E402

from dgx_monarch.actor import store_detect  # noqa: E402
from dgx_monarch.actor.unbake import RESTORABLE_QUANT_FORMATS  # noqa: E402
from dgx_monarch.adapters import ADAPTERS  # noqa: E402
from dgx_monarch.adapters.base import UnsupportedModelError  # noqa: E402
from dgx_monarch.adapters.detect import (  # noqa: E402
    detect_family_from_header,
    detect_quant_from_header,
)
from dgx_monarch.adapters.fsdp import FSDP_ADMITTED_CHECKPOINT_KINDS, validate_fsdp_launch_quant  # noqa: E402
from dgx_monarch.lora_quant_guard import lora_refused  # noqa: E402
from dgx_monarch.topology import choose_auto_topology  # noqa: E402

LINEARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
SHAPES = {"attn.qkv_proj": [21504, 5376], "attn.out_proj": [5376, 7168],
          "mlp.fc1": [28672, 5376], "mlp.fc2": [5376, 14336]}


def _header(kind: str) -> dict:
    """One p-h3-image DiT header of the given file kind, block 0 and the shared rows."""
    tensors: dict[str, dict] = {
        "adaln_t_table": {"dtype": "F32", "shape": [4097, 16]},
        "video_patch_proj.weight": {"dtype": "F32", "shape": [5376, 96]},
        "audio_patch_proj.weight": {"dtype": "F32", "shape": [5376, 32]},
        "condition_proj.weight": {"dtype": "BF16", "shape": [5376, 5120]},
        "blocks.0.adaln_proj.linear.weight": {"dtype": "F32", "shape": [96768, 16]},
        "blocks.0.attn.q_norm.weight": {"dtype": "BF16", "shape": [128]},
    }
    for name in LINEARS:
        rows, cols = SHAPES[name]
        key = f"blocks.0.{name}"
        if kind == "bf16":
            tensors[f"{key}.weight"] = {"dtype": "BF16", "shape": [rows, cols]}
        elif kind == "int8":
            tensors[f"{key}.weight"] = {"dtype": "I8", "shape": [rows, cols]}
            tensors[f"{key}.weight_scale"] = {"dtype": "F32", "shape": [rows, 1]}
            tensors[f"{key}.comfy_quant"] = {"dtype": "U8", "shape": [72]}
        else:  # w4a8: two int4 codes per byte, an FP8 scale per 16 inputs
            tensors[f"{key}.weight"] = {"dtype": "I8", "shape": [rows, cols // 2]}
            tensors[f"{key}.weight_codebook"] = {"dtype": "F32", "shape": [16]}
            tensors[f"{key}.weight_s_channel"] = {"dtype": "F32", "shape": [rows]}
            tensors[f"{key}.weight_s_rel"] = {"dtype": "F8_E4M3", "shape": [rows, cols // 16]}
            tensors[f"{key}.comfy_quant"] = {"dtype": "U8", "shape": [89]}
    # The token refiner stays int8_tensorwise in the w4a8 file.
    refiner = "token_refiner.blocks.0.attn.qkv_proj"
    if kind == "bf16":
        tensors[f"{refiner}.weight"] = {"dtype": "BF16", "shape": [21504, 5376]}
    else:
        tensors[f"{refiner}.weight"] = {"dtype": "I8", "shape": [21504, 5376]}
        tensors[f"{refiner}.weight_scale"] = {"dtype": "F32", "shape": [21504, 1]}
    return {"__tensors__": tensors}


@pytest.mark.parametrize("kind, expected", [("bf16", "bf16"), ("int8", "int8"), ("w4a8", "w4a8")])
def test_every_release_file_reads_as_minimax_h3_with_its_own_kind(kind, expected):
    header = _header(kind)
    keys = list(header["__tensors__"])
    assert detect_family_from_header(keys, header) == "minimax_h3"
    assert detect_quant_from_header(keys, header) == expected


def test_one_adapter_owns_the_family():
    assert [adapter.family for adapter in ADAPTERS].count("minimax_h3") == 1


def test_an_fp8_file_with_scales_still_reads_fp8():
    """Only the grouped 4-bit sidecar turns F8 scales into w4a8."""
    tensors = {"blocks.0.mlp.fc1.weight": {"dtype": "F8_E4M3"},
               "blocks.0.mlp.fc1.weight_scale": {"dtype": "F32"},
               "blocks.0.mlp.fc2.weight": {"dtype": "I8"}}
    assert detect_quant_from_header(list(tensors), {"__tensors__": tensors}) == "fp8"


class _Linear:
    def __init__(self, fmt):
        self.quant_format = fmt


class _Model:
    def __init__(self, formats):
        self._modules = [_Linear(fmt) for fmt in formats]

    def modules(self):
        return iter([SimpleNamespace(), *self._modules])

    def named_parameters(self):
        return iter(())


def _live(formats, caplog=None):
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=_Model(formats), model_config=None))
    return store_detect._detect_live_precision(patcher, "bf16")


def test_live_detection_names_the_w4a8_file_whatever_layer_comes_first():
    # The token refiner, the first quantized module, stays int8_tensorwise.
    evidence = _live(["int8_tensorwise", "int8_tensorwise", "asym_w4a8_int8", "int8_tensorwise"])
    assert (evidence.quant_kind, evidence.quantized_shards) == ("w4a8", True)
    assert _live(["int8_tensorwise", "int8_tensorwise"]).quant_kind == "int8"
    assert _live(["float8_e4m3fn", "nvfp4"]).quant_kind == "fp8"


@pytest.mark.parametrize("fmt", ["w3a8_int8", "asym_w4a8_int8", "w5a8_int8", "w6a8_int8"])
def test_every_grouped_format_reads_w4a8_live_as_its_header_does(fmt, caplog):
    """ComfyUI's grouped formats all store ``weight_s_rel`` group scales, which
    is what the header reading names w4a8, so a file of any of them reads the
    same once loaded and never warns as an unknown format."""
    tensors = {"blocks.0.fc.weight": {"dtype": "I8"}, "blocks.0.fc.weight_s_rel": {"dtype": "F8_E4M3"},
               "blocks.0.fc.weight_s_channel": {"dtype": "F32"}, "blocks.0.fc.comfy_quant": {"dtype": "U8"}}
    assert detect_quant_from_header(list(tensors), {"__tensors__": tensors}) == "w4a8"
    with caplog.at_level("WARNING"):
        evidence = _live(["int8_tensorwise", fmt, "int8_tensorwise"])
        assert (evidence.quant_kind, evidence.quantized_shards) == ("w4a8", True)
        assert _live([fmt]).quant_kind == "w4a8"
    assert not any("unknown quant_format" in record.message for record in caplog.records)


def test_a_convrot_w4a4_file_reads_its_own_kind_whatever_layer_comes_first(caplog):
    """ConvRot W4A4 runs comfy-kitchen's int4 GEMM, which no render here has
    measured. Its header reads int8, and no header key tells it from an INT8
    file, so the loaded reading names it: w4a4, never the int8 that holds
    sharded evidence, and never an unknown format called fp8."""
    tensors = {"blocks.0.fc.weight": {"dtype": "I8"}, "blocks.0.fc.weight_scale": {"dtype": "F32"},
               "blocks.0.fc.comfy_quant": {"dtype": "U8"}}
    assert detect_quant_from_header(list(tensors), {"__tensors__": tensors}) == "int8"
    with caplog.at_level("WARNING"):
        for formats in (["convrot_w4a4"], ["int8_tensorwise", "convrot_w4a4", "int8_tensorwise"]):
            evidence = _live(formats)
            assert (evidence.quant_kind, evidence.quantized_shards) == ("w4a4", True)
    assert not any("unknown quant_format" in record.message for record in caplog.records)
    # The loaded reading goes before the header's, so the Worker holds w4a4.
    assert store_detect._combine_checkpoint_and_live_kinds(
        "int8", "w4a4", live_dtypes_are_the_file=True,
        live_profile=evidence.live_dtype_profile) == "w4a4"
    assert store_detect._is_quantized_kind("w4a4")


@pytest.mark.parametrize("fmt", ["w3a8_int8", "w5a8_int8", "w6a8_int8", "convrot_w4a4"])
def test_the_kinds_these_formats_now_read_take_no_other_kinds_evidence(fmt):
    """What a Worker does with the kind, for the formats whose reading changed:
    a family that lists the kinds it measured under sharding refuses both new
    readings, MOSS-Image refuses a LoRA on them as unmeasured, and FSDP
    refuses them by kind."""
    from dgx_monarch.adapters import cfg_parallel
    from dgx_monarch.adapters.moss_image import MOSSImageAdapter

    kind = _live(["int8_tensorwise", fmt]).quant_kind
    assert kind == ("w4a4" if fmt == "convrot_w4a4" else "w4a8")
    assert kind not in MOSSImageAdapter.sp_validated_quants
    adapter = MOSSImageAdapter()
    for sp, cfg in ((2, 1), (1, 2)):
        with pytest.raises(UnsupportedModelError) as refused:
            cfg_parallel.assert_sp_quant_supported(adapter, sp, kind, cfg=cfg)
        words = str(refused.value)
        # The family's own reason, with the kind the Worker read named ahead of it.
        assert f"the loaded file reads as quant={kind}" in words, words
        assert "moss_image shards its joint stream" in words
        assert "mode=local with gpus_per_host=1" in words
    assert "reads an unrecognized" not in adapter.sp_quant_refusal_reason
    cfg_parallel.assert_sp_quant_supported(adapter, 1, kind)  # one GPU runs ComfyUI's own forward
    cfg_parallel.assert_sp_quant_supported(SimpleNamespace(family="minimax_h3"), 2, kind)  # no list, no gate
    # No render has measured a LoRA on these kinds on MOSS-Image, whose INT8
    # file with one rendered wrong, so the pair refuses on every topology, in
    # words that claim no measurement of its own; other families stay open.
    from dgx_monarch.lora_quant_guard import assert_lora_quant_supported

    stack = [{"name": "some.safetensors", "strength": 1.0}]
    assert lora_refused("moss_image", kind, stack) and not lora_refused("moss_image", kind, [])
    assert not lora_refused("minimax_h3", kind, stack)
    with pytest.raises(UnsupportedModelError) as caught:
        assert_lora_quant_supported("moss_image", kind, stack)
    text = str(caught.value)
    assert text.startswith(f"[dgxm:P] moss_image: no render has measured a LoRA (some.safetensors) "
                           f"on a {kind.upper()} file of this family."), text
    assert f"{kind.upper()} file with a LoRA (some.safetensors) rendered far" not in text
    assert "until a render measures it" in text and "mode=local on one GPU included" in text
    assert "Render this graph in stock ComfyUI on one GPU instead" in text
    assert text.endswith("See docs/TROUBLESHOOTING.md #129.")
    assert kind not in FSDP_ADMITTED_CHECKPOINT_KINDS
    with pytest.raises(UnsupportedModelError, match="have no registered shard layout"):
        validate_fsdp_launch_quant(kind)


def test_an_unknown_format_still_reads_fp8_and_warns_once(caplog):
    with caplog.at_level("WARNING"):
        assert _live(["w9a9", "w9a9", "w9a9"]).quant_kind == "fp8"
    assert sum("unknown quant_format" in record.message for record in caplog.records) == 1


def test_fsdp_refuses_the_w4a8_file_before_it_builds_a_shard():
    assert "w4a8" not in FSDP_ADMITTED_CHECKPOINT_KINDS
    with pytest.raises(UnsupportedModelError, match="the grouped INT8 layouts and ConvRot W4A4 have no registered"):
        validate_fsdp_launch_quant("w4a8")


def test_a_lora_change_on_the_w4a8_file_reloads_rather_than_unbakes():
    """The lazy un-bake restores int8_tensorwise and the fp8 and fp4 layouts only."""
    assert "int8_tensorwise" in RESTORABLE_QUANT_FORMATS
    assert not {"w3a8_int8", "asym_w4a8_int8", "w5a8_int8", "w6a8_int8",
                "convrot_w4a4"} & RESTORABLE_QUANT_FORMATS


@pytest.mark.parametrize("quant", ["bf16", "int8", "w4a8"])
def test_auto_takes_the_h3_row_for_every_file_and_no_lora_refusal_applies(quant):
    topology = choose_auto_topology("minimax_h3", quant, 8.48, 2, cfg_value=1.0).topology
    assert (topology.ulysses, topology.cfg, topology.dp) == (2, 1, 1)
    assert not lora_refused("minimax_h3", quant, [{"name": "p-h3-image_lora_bf16.safetensors"}])


def test_the_three_dit_files_share_one_sweep_stem():
    stems = {split_quant(name) for name in
             ("p-h3-image_bf16", "p-h3-image_int8_convrot", "p-h3-image_w4a8")}
    assert stems == {("ph3image", "bf16"), ("ph3image", "int8_convrot"), ("ph3image", "w4a8")}


def test_the_converter_leaves_a_forced_input_out_of_the_widgets():
    """TextGenerateV2's system_prompt is a socket, so it holds no widgets_values slot."""
    inputs = {"required": {"clip": ["CLIP"], "prompt": ["STRING", {"multiline": True}],
                           "max_length": ["INT", {"default": 512}],
                           "sampling_mode": ["COMFY_DYNAMICCOMBO_V3", {}]},
              "optional": {"thinking": ["BOOLEAN", {}], "use_default_template": ["BOOLEAN", {}],
                           "mtp": [["auto", "off"], {}],
                           "system_prompt": ["STRING", {"forceInput": True}]}}
    assert convert.object_info_widget_names(inputs) == [
        "prompt", "max_length", "sampling_mode", "thinking", "use_default_template", "mtp"]
