"""tools/make_pin_probe_lora.py on synthetic safetensors headers (issue #30).

The tool reads a checkpoint's header, finds one family's attention modules
from their query projection keys, and writes a LoRA holding only
``diffusion_model.<module>.comfy_attention.config`` tensors. Canary seam 39
runs its patterns against tiny real ComfyUI models and its LoRA through
ComfyUI's own patcher; these tests cover the file handling.
"""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_pin_probe_lora.py"
SPEC = importlib.util.spec_from_file_location("make_pin_probe_lora", TOOL)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

HEADERS = {
    "krea2": ["blocks.{n}.attn.wq.weight", "blocks.{n}.attn.wk.weight",
              "txtfusion.layerwise_blocks.{n}.attn.wq.weight",
              "txtfusion.refiner_blocks.{n}.attn.wq.weight", "blocks.{n}.mlp.w1.weight"],
    "anima": ["net.blocks.{n}.self_attn.q_proj.weight", "net.blocks.{n}.cross_attn.q_proj.weight",
              "net.llm_adapter.blocks.{n}.self_attn.q_proj.weight", "net.x_embedder.{n}.weight"],
    "kandinsky5": ["text_transformer_blocks.{n}.self_attention.to_query.weight",
                   "visual_transformer_blocks.{n}.self_attention.to_query.weight",
                   "visual_transformer_blocks.{n}.cross_attention.to_query.weight"],
    "wan21": ["model.diffusion_model.blocks.{n}.self_attn.q.weight",
              "model.diffusion_model.blocks.{n}.self_attn.q.scale_weight",
              "model.diffusion_model.blocks.{n}.cross_attn.q.weight",
              "model.diffusion_model.head.head.{n}.weight"],
    "qwen_image21": ["transformer_blocks.{n}.attn.to_q.weight",
                     "transformer_blocks.{n}.attn.to_k.weight"],
    # FLUX.3: the int8 files fuse qkv, the bf16 and nvfp4 files split it; a
    # header mixing both spellings, with the int8 scale and pin beside them.
    "flux3": ["single_blocks.{n}.qkv.weight", "single_blocks.{n}.qkv.weight_scale",
              "single_blocks.{n}.mlp_in.weight", "txt_mode_blocks.{n}.q_proj.weight",
              "txt_mode_blocks.{n}.k_proj.weight", "content_mode_blocks.image.{n}.qkv.weight",
              "content_mode_blocks.image.{n}.comfy_attention.config",
              "content_mode_blocks.image_cond.{n}.q_proj.weight"],
}
HEADERS["flux3_early"] = HEADERS["flux3"]

EXPECTED_SELF = {
    "krea2": ["blocks.{n}.attn", "txtfusion.layerwise_blocks.{n}.attn",
              "txtfusion.refiner_blocks.{n}.attn"],
    "anima": ["blocks.{n}.self_attn"],
    "kandinsky5": ["text_transformer_blocks.{n}.self_attention",
                   "visual_transformer_blocks.{n}.self_attention"],
    "wan21": ["blocks.{n}.self_attn"],
    "qwen_image21": ["transformer_blocks.{n}.attn"],
    "flux3": ["single_blocks.{n}", "txt_mode_blocks.{n}", "content_mode_blocks.image.{n}",
              "content_mode_blocks.image_cond.{n}"],
    "flux3_early": ["txt_mode_blocks.{n}", "content_mode_blocks.image_cond.{n}"],
}


def _checkpoint(tmp_path: Path, family: str, blocks: int = 12) -> Path:
    tensors = {pattern.format(n=n): torch.zeros(1)
               for pattern in HEADERS[family] for n in range(blocks)}
    path = tmp_path / f"{family}.safetensors"
    save_file(tensors, str(path))
    return path


def _expected(family: str, blocks: int = 12) -> list[str]:
    paths = [pattern.format(n=n) for pattern in EXPECTED_SELF[family] for n in range(blocks)]
    return sorted(paths, key=lambda p: tuple(int(x) if x.isdigit() else x for x in p.split(".")))


@pytest.mark.parametrize("family", sorted(HEADERS))
def test_each_family_writes_one_config_per_self_attention_module(tmp_path, family):
    out = tmp_path / "pin.safetensors"
    code = probe.main([str(_checkpoint(tmp_path, family)), str(out), "--family", family,
                       "--kernel", "comfy_kitchen_sol", "--tau", "0.7"])
    assert code == 0
    lora = load_file(str(out))
    assert sorted(lora) == sorted(
        f"diffusion_model.{path}.comfy_attention.config" for path in _expected(family))
    for tensor in lora.values():
        # The encoding ComfyAttention._load_from_state_dict decodes.
        assert tensor.dtype is torch.uint8
        assert json.loads(tensor.numpy().tobytes()) == {"attention": "comfy_kitchen_sol",
                                                        "tau": 0.7}


def test_blocks_come_in_numeric_order_and_can_be_limited(tmp_path):
    keys = probe.read_header_keys(_checkpoint(tmp_path, "wan21"))
    paths = probe.attention_paths("wan21", keys)
    assert paths[:3] == ["blocks.0.self_attn", "blocks.1.self_attn", "blocks.2.self_attn"]
    assert paths[-1] == "blocks.11.self_attn"
    out = tmp_path / "four.safetensors"
    assert probe.main([str(_checkpoint(tmp_path, "wan21")), str(out), "--family", "wan21",
                       "--kernel", "comfy_kitchen_int8", "--blocks", "4"]) == 0
    lora = load_file(str(out))
    assert len(lora) == 4
    assert {json.loads(t.numpy().tobytes())["attention"] for t in lora.values()} == {
        "comfy_kitchen_int8"}


def test_cross_attention_joins_only_on_request_and_never_for_sol(tmp_path):
    keys = probe.read_header_keys(_checkpoint(tmp_path, "kandinsky5", blocks=2))
    assert len(probe.attention_paths("kandinsky5", keys)) == 4
    assert len(probe.attention_paths("kandinsky5", keys, cross=True)) == 6
    with pytest.raises(SystemExit):
        probe.main([str(_checkpoint(tmp_path, "kandinsky5")), str(tmp_path / "x.safetensors"),
                    "--family", "kandinsky5", "--kernel", "comfy_kitchen_sol", "--cross"])


def test_a_checkpoint_of_another_family_matches_nothing_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "none.safetensors"
    assert probe.main([str(_checkpoint(tmp_path, "qwen_image21")), str(out),
                       "--family", "wan21", "--kernel", "comfy_kitchen_int8"]) == 1
    assert not out.exists()
    assert "no wan21 attention keys matched" in capsys.readouterr().err


def test_an_existing_output_is_never_overwritten(tmp_path):
    out = tmp_path / "taken.safetensors"
    out.write_bytes(b"keep")
    assert probe.main([str(_checkpoint(tmp_path, "wan21")), str(out), "--family", "wan21",
                       "--kernel", "comfy_kitchen_int8"]) == 2
    assert out.read_bytes() == b"keep"


def test_tau_belongs_to_sol_and_unknown_kernels_refuse():
    assert probe.pin_config("comfy_kitchen_sol", None) == {"attention": "comfy_kitchen_sol",
                                                           "tau": 1.0}
    with pytest.raises(ValueError, match="comfy_kitchen_sol only"):
        probe.pin_config("comfy_kitchen_int8", 0.5)
    with pytest.raises(ValueError, match="unknown kernel"):
        probe.pin_config("flash", None)


def test_a_file_that_is_not_safetensors_refuses_cleanly(tmp_path, capsys):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", 1 << 40))
    assert probe.main([str(bad), str(tmp_path / "o.safetensors"), "--family", "wan21",
                       "--kernel", "comfy_kitchen_int8"]) == 2
    assert "header" in capsys.readouterr().err


def test_the_tool_stays_small_and_needs_only_torch_and_safetensors():
    source = TOOL.read_text(encoding="utf-8")
    assert len(source.splitlines()) < 200
    imports = {line.split()[1].split(".")[0] for line in source.splitlines()
               if line.strip().startswith(("import ", "from ")) and "__future__" not in line}
    assert imports <= {"argparse", "json", "re", "struct", "sys", "pathlib", "torch",
                       "safetensors"}


def test_flux3_early_is_the_early_subset_and_a_fused_or_split_file_reads_alike(tmp_path):
    """The early pattern is the ten stock sites comfy runs from preprocess_conds,
    a subset of the family pattern, and the fused int8 spelling and the split
    bf16 spelling name the same blocks."""
    keys = probe.read_header_keys(_checkpoint(tmp_path, "flux3", blocks=2))
    every, early = probe.attention_paths("flux3", keys), probe.attention_paths("flux3_early", keys)
    assert set(early) < set(every)
    assert not any(path.startswith(("single_blocks.", "content_mode_blocks.image."))
                   for path in early)
    fused = ["model.diffusion_model." + key for key in
             ("single_blocks.0.qkv.weight", "txt_mode_blocks.0.qkv.weight")]
    split = ["model.diffusion_model." + key for key in
             ("single_blocks.0.q_proj.weight", "txt_mode_blocks.0.q_proj.weight")]
    filler = [f"model.diffusion_model.single_blocks.0.x{i}" for i in range(6)]
    assert probe.attention_paths("flux3", fused + filler) == probe.attention_paths(
        "flux3", split + filler) == ["single_blocks.0", "txt_mode_blocks.0"]
