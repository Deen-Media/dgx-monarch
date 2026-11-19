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


def test_an_unknown_format_still_reads_fp8_and_warns_once(caplog):
    with caplog.at_level("WARNING"):
        assert _live(["w9a9", "w9a9", "w9a9"]).quant_kind == "fp8"
    assert sum("unknown quant_format" in record.message for record in caplog.records) == 1


def test_fsdp_refuses_the_w4a8_file_before_it_builds_a_shard():
    assert "w4a8" not in FSDP_ADMITTED_CHECKPOINT_KINDS
    with pytest.raises(UnsupportedModelError, match="grouped 4-bit layouts have no registered shard"):
        validate_fsdp_launch_quant("w4a8")


def test_a_lora_change_on_the_w4a8_file_reloads_rather_than_unbakes():
    """The lazy un-bake restores int8_tensorwise and the fp8 and fp4 layouts only."""
    assert "int8_tensorwise" in RESTORABLE_QUANT_FORMATS
    assert "asym_w4a8_int8" not in RESTORABLE_QUANT_FORMATS


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
