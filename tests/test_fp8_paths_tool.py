"""tools/fp8_paths.py: which operations class ComfyUI picks for a checkpoint, by header.

Fixtures are small real safetensors files: real dtypes and shapes, zero
payload except the per-layer ``comfy_quant`` records, which hold their JSON as
ComfyUI writes it. The last case checks the reproduced rule against a real
ComfyUI checkout when ``COMFYUI_DIR`` names one (the comfy canary sets it).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path

import pytest

from dgx_monarch.safetensors_header import DTYPE_BITS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fp8_paths", ROOT / "tools" / "fp8_paths.py")
assert SPEC is not None and SPEC.loader is not None
fp8_paths = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("fp8_paths", fp8_paths)
SPEC.loader.exec_module(fp8_paths)

PLAIN, FAST, LOADER, BOTH = (s[0] for s in fp8_paths.SETTINGS)


def _write(path: Path, tensors: dict[str, tuple[str, list[int]] | bytes],
           metadata: dict[str, str] | None = None) -> Path:
    """A safetensors file; a bytes value is a 1-D U8 tensor holding those bytes."""
    header: dict[str, object] = {}
    payload = bytearray()
    for key, spec in tensors.items():
        if isinstance(spec, bytes):
            dtype, shape, data = "U8", [len(spec)], spec
        else:
            dtype, shape = spec
            data = b"\0" * (math.prod(shape) * DTYPE_BITS[dtype] // 8)
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(payload))
    return path


def _picks(path: Path, fp8_compute: bool = True) -> dict[str, object]:
    facts = fp8_paths.read_facts(path)
    assert not facts.error, facts.error
    return {name: fp8_paths.predict(facts, fast, loader, fp8_compute)
            for name, fast, loader in fp8_paths.SETTINGS}


def _record(conf: dict) -> bytes:
    return json.dumps(conf).encode()


def test_a_plain_fp8_file_dequantizes_unless_a_flag_or_the_loader_asks(tmp_path):
    path = _write(tmp_path / "plain_fp8.safetensors", {
        "transformer_blocks.0.attn.to_q.weight": ("F8_E4M3", [64, 64]),
        "transformer_blocks.0.attn.to_k.weight": ("F8_E4M3", [64, 64]),
        "img_in.bias": ("F8_E4M3", [64]),
    })
    facts = fp8_paths.read_facts(path)
    assert (facts.kitchen, facts.weight_dtype) == (False, "float8_e4m3fn")
    picks = _picks(path)
    assert picks[PLAIN].operations == "manual_cast"
    assert "dequantized" in picks[PLAIN].matmul
    for name in (FAST, LOADER, BOTH):
        assert picks[name].operations == "fp8_ops"
        assert picks[name].matmul.startswith("torch._scaled_mm")
        assert picks[name].notes == []


def test_without_fp8_compute_nothing_takes_fp8_ops(tmp_path):
    path = _write(tmp_path / "plain_fp8.safetensors", {"blocks.0.w.weight": ("F8_E4M3", [32, 32])})
    picks = _picks(path, fp8_compute=False)
    assert {pick.operations for pick in picks.values()} == {"manual_cast"}


def test_a_bf16_file_takes_fp8_ops_under_the_flag_but_keeps_its_matmul(tmp_path):
    path = _write(tmp_path / "full.safetensors", {
        "model.diffusion_model.blocks.0.w.weight": ("BF16", [64, 64]),
        "model.diffusion_model.blocks.0.norm.weight": ("F32", [64]),
    })
    picks = _picks(path)
    assert picks[PLAIN].operations == "disable_weight_init"
    assert picks[FAST].operations == "fp8_ops"
    assert picks[FAST].matmul.startswith("cast path")
    # The loader's fast dtype casts the weights to FP8 first, then scaled_mm runs.
    assert picks[LOADER].operations == "fp8_ops"
    assert picks[LOADER].matmul.startswith("torch._scaled_mm")
    assert "casts bfloat16 weights to float8_e4m3fn" in picks[LOADER].notes[0]


def test_header_quantization_metadata_takes_kitchen_ops_whatever_the_flags(tmp_path):
    layers = {"blocks.0.wq": {"format": "float8_e4m3fn"},
              "blocks.0.wo": {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}}
    path = _write(tmp_path / "scaled.safetensors", {
        "blocks.0.wq.weight": ("F8_E4M3", [64, 64]),
        "blocks.0.wq.weight_scale": ("F32", []),
        "blocks.0.wo.weight": ("F8_E4M3", [64, 64]),
        "blocks.0.wo.weight_scale": ("F32", []),
        "blocks.0.norm.weight": ("BF16", [64]),
    }, metadata={"_quantization_metadata": json.dumps({"layers": layers})})
    facts = fp8_paths.read_facts(path)
    assert facts.kitchen and facts.sources == ["header _quantization_metadata"]
    assert facts.formats == {"float8_e4m3fn": 2} and facts.full_precision_layers == 1
    assert fp8_paths.kitchen_label(facts) == "fp8_scaled, 1 of 2 layers full precision"
    picks = _picks(path)
    assert {pick.operations for pick in picks.values()} == {"mixed_precision_ops"}
    assert any("full_precision_matrix_mult" in note for note in picks[PLAIN].notes)
    assert any("does not quantize" in note for note in picks[LOADER].notes)


def test_a_legacy_marker_with_two_elements_makes_every_layer_full_precision(tmp_path):
    path = _write(tmp_path / "legacy.safetensors", {
        "scaled_fp8": ("F8_E4M3", [2]),
        "blocks.0.self_attn.q.weight": ("F8_E4M3", [64, 64]),
        "blocks.0.self_attn.q.scale_weight": ("F32", []),
        "blocks.0.ffn.0.weight": ("F8_E4M3", [64, 64]),
        "blocks.0.ffn.0.scale_weight": ("F32", []),
    })
    facts = fp8_paths.read_facts(path)
    assert facts.kitchen and facts.sources == ["legacy scaled_fp8 marker"]
    assert facts.formats == {"float8_e4m3fn": 2} and facts.full_precision_layers == 2


def test_direct_per_layer_records_are_read_and_name_int8_convrot(tmp_path):
    conf = {"format": "int8_tensorwise", "convrot": True}
    path = _write(tmp_path / "convrot.safetensors", {
        "model.diffusion_model.blocks.0.mlp.weight": ("I8", [64, 64]),
        "model.diffusion_model.blocks.0.mlp.weight_scale": ("F32", [64]),
        "model.diffusion_model.blocks.0.mlp.comfy_quant": _record(conf),
        "model.diffusion_model.blocks.0.norm.weight": ("BF16", [64]),
    })
    facts = fp8_paths.read_facts(path)
    assert facts.kitchen and facts.sources == ["per-layer comfy_quant tensors"]
    assert fp8_paths.kitchen_label(facts) == "int8_convrot"
    assert facts.weight_dtype == "int8"


def test_a_record_outside_the_unet_prefix_reaches_no_layer(tmp_path):
    """ComfyUI strips the unet prefix and drops every key outside it."""
    tensors = {f"model.diffusion_model.blocks.{i}.w.weight": ("F8_E4M3", [32, 32]) for i in range(6)}
    tensors["first_stage_model.decoder.w.comfy_quant"] = _record({"format": "float8_e4m3fn"})
    facts = fp8_paths.read_facts(_write(tmp_path / "outside.safetensors", tensors))
    assert facts.kitchen is False
    assert _picks(tmp_path / "outside.safetensors")[PLAIN].operations == "manual_cast"


def test_the_cli_prints_names_not_paths_and_survives_a_bad_file(tmp_path, capsys):
    models = tmp_path / "models"
    (models / "nested").mkdir(parents=True)
    _write(models / "nested" / "a_fp8.safetensors", {"w.weight": ("F8_E4M3", [8, 8])})
    (models / "broken.safetensors").write_bytes(b"not a header")
    (models / "notes.txt").write_text("skip me")
    assert fp8_paths.main(["--json", str(models)]) == 0
    out = capsys.readouterr().out
    assert str(tmp_path) not in out
    files = {row["name"]: row for row in json.loads(out)["files"]}
    assert set(files) == {"a_fp8.safetensors", "broken.safetensors"}
    assert files["broken.safetensors"]["error"]
    assert files["a_fp8.safetensors"]["picks"][FAST]["operations"] == "fp8_ops"
    assert fp8_paths.main([str(models)]) == 0
    table = capsys.readouterr().out
    assert "| a_fp8.safetensors |" in table and str(tmp_path) not in table


def _fake_comfy(root: Path, *, drop: str = "") -> Path:
    """A checkout holding just the functions the rule check reads."""
    sources = {
        "comfy/ops.py": '''
def fp8_linear(self, input):
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False,
                    fp8_optimizations=False, model_config=None):
    if model_config and hasattr(model_config, 'quant_config') and model_config.quant_config:
        return mixed_precision_ops(model_config.quant_config, compute_dtype)
    if fp8_compute and (fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast):
        return fp8_ops
    if compute_dtype is None or weight_dtype == compute_dtype:
        return disable_weight_init
    return manual_cast
''',
        "nodes.py": '''
class UNETLoader:
    def load_unet(self, unet_name, weight_dtype):
        model_options["fp8_optimizations"] = True
''',
        "comfy/sd.py": '''
def load_diffusion_model_state_dict(sd, model_options={}):
    if model_options.get("fp8_optimizations", False):
        pass
''',
        "comfy/model_management.py": '''
def unet_dtype(device=None):
    if supports_fp8_compute(device):
        return fp8_dtype
''',
        "comfy/utils.py": '''
def detect_layer_quantization(state_dict, prefix):
    return any(k.endswith(".comfy_quant") for k in state_dict)
''',
        "comfyui_version.py": '__version__ = "9.9.9"\n',
    }
    for relative, text in sources.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.replace(drop, "") if drop else text)
    return root


def test_the_rule_check_names_a_clause_a_checkout_dropped(tmp_path, capsys):
    holds = _fake_comfy(tmp_path / "holds")
    assert fp8_paths.check_rule(holds) == []
    assert fp8_paths.comfy_version(holds) == "9.9.9"
    gone = _fake_comfy(tmp_path / "gone", drop="fp8_optimizations or ")
    missing = fp8_paths.check_rule(gone)
    assert missing == ["comfy/ops.py pick_operations: fp8_optimizations or "
                       "PerformanceFeature.Fp8MatrixMultiplication in args.fast"]
    model = _write(tmp_path / "m.safetensors", {"w.weight": ("F8_E4M3", [8, 8])})
    assert fp8_paths.main(["--comfy-dir", str(gone), str(model)]) == 1
    assert "these clauses are gone" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("COMFYUI_DIR"), reason="needs a ComfyUI checkout (COMFYUI_DIR)")
def test_the_reproduced_rule_holds_on_the_checked_out_comfyui():
    """The comfy canary runs this against master and the newest release tag."""
    comfy = Path(os.environ["COMFYUI_DIR"]).expanduser()
    assert fp8_paths.check_rule(comfy) == [], (
        "ComfyUI changed how it picks a model's operations class; re-read "
        "comfy/ops.py pick_operations and update tools/fp8_paths.py and "
        "docs/MODELS.md, FP8 compute paths")
