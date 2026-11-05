"""The p-h3-image template keeps the vendor's recipe, and its testing graphs carry the matching sets.

The vendor's own text-to-image workflow (PrunaAI/P-H3-Image-Comfy at
44a0fb73f299, workflows/p-h3-image_text_to_image.json) loads the BF16 DiT,
text encoder, VAE and prompt upsampler, sets ModelSamplingMiniMaxH3 to 1.0
and 1.0, and samples the 8-step grid below with euler through a Basic Guider
and Sampler Custom Advanced at CFG 1, on an 8 MP square. The converted API
graph must keep each of those, with the DGX Monarch carriers in place of the
stock model nodes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The comfy-canary job runs plain pytest, so the checkout root goes on sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sweep import convert  # noqa: E402
from benchmark.sweep.matrix import (  # noqa: E402
    GRAPH_LORA_STRENGTH_TEMPLATES,
    LORA_CELL_STRENGTH,
    graph_facts,
    lora_stack,
)

NAME = "dgx-monarch-p-h3-image"
GENERATED = ROOT / "tests" / "fixtures" / "workflows" / "generated"
SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"
FILES = {
    "bf16": ("p-h3-image_bf16.safetensors", "p-h3-image_text_encoder_bf16.safetensors",
             "p-h3-image_vae_bf16.safetensors"),
    "int8": ("p-h3-image_int8_convrot.safetensors", "p-h3-image_text_encoder_int8_convrot.safetensors",
             "p-h3-image_vae_int8_convrot.safetensors"),
    "w4a8": ("p-h3-image_w4a8.safetensors", "p-h3-image_text_encoder_w4a8.safetensors",
             "p-h3-image_vae_int8_convrot.safetensors"),
    "lora": ("minimax_h3_fl2va_pruned_int8_convrot.safetensors",
             "p-h3-image_text_encoder_int8_convrot.safetensors",
             "p-h3-image_vae_int8_convrot.safetensors"),
}
PATHS = {"bf16": ROOT / "example_workflows" / f"{NAME}.json",
         "int8": GENERATED / "dgx-monarch-test-p-h3-image-int8.json",
         "w4a8": GENERATED / "dgx-monarch-test-p-h3-image-w4a8.json",
         "lora": GENERATED / "dgx-monarch-test-p-h3-image-lora.json"}


def _graph(kind="bf16"):
    return convert.convert(json.loads(PATHS[kind].read_text()), convert.repo_widget_names(ROOT))


def _one(graph, kind):
    found = [(key, node["inputs"]) for key, node in graph.items() if node["class_type"] == kind]
    assert len(found) == 1, kind
    return found[0]


def test_the_template_keeps_the_vendor_recipe():
    graph = _graph()
    init_id, init = _one(graph, "DGXMonarchInit")
    unet_id, unet = _one(graph, "DGXMonarchUNETLoader")
    shift_id, shift = _one(graph, "DGXMonarchModelSamplingMiniMaxH3")
    guider_id, guider = _one(graph, "DGXMonarchBasicGuider")
    cond_id, cond = _one(graph, "PH3ImageConditioning")
    _, sampler = _one(graph, "KSamplerSelect")
    _, sigmas = _one(graph, "ManualSigmas")
    _, custom = _one(graph, "DGXMonarchSamplerCustom")
    assert (init["topology"], init["mode"], init["auto_gate"]) == ("auto", "auto", "first_use")
    assert (unet["unet_name"], unet["weight_dtype"], unet["mesh"]) == (FILES["bf16"][0], "default", [init_id, 0])
    assert (shift["shift_video"], shift["shift_audio"], shift["model"]) == (1.0, 1.0, [unet_id, 0])
    assert guider["model"] == [shift_id, 0] and guider["conditioning"] == [cond_id, 0]
    assert (cond["width"], cond["height"]) == (2912, 2912)
    assert (sampler["sampler_name"], sigmas["sigmas"]) == ("euler", SIGMAS)
    assert custom["guider"] == [guider_id, 0] and custom["latent_image"] == [cond_id, 1]
    clips = {key: node["inputs"] for key, node in graph.items() if node["class_type"] == "CLIPLoader"}
    assert {(c["clip_name"], c["type"]) for c in clips.values()} == {
        (FILES["bf16"][1], "minimax"), ("p-h3-image_prompt_upsampler_bf16.safetensors", "minimax")}
    assert graph[cond["clip"][0]]["inputs"]["clip_name"] == FILES["bf16"][1]
    assert graph[cond["vae"][0]]["inputs"]["vae_name"] == FILES["bf16"][2]


def test_the_upsampler_feeds_the_prompt_through_the_switch_and_the_preview():
    graph = _graph()
    _, cond = _one(graph, "PH3ImageConditioning")
    _, preview = _one(graph, "PreviewAny")
    switch_id, switch = _one(graph, "ComfySwitchNode")
    upsample_id, upsample = _one(graph, "TextGenerateV2")
    request_id, request = _one(graph, "PrimitiveStringMultiline")
    assert cond["prompt"] == [switch_id, 0] and preview["source"] == [switch_id, 0]
    assert switch["switch"] is True
    assert switch["on_true"] == [upsample_id, 0] and switch["on_false"] == [request_id, 0]
    assert upsample["prompt"] == [request_id, 0]
    # Greedy decoding with thinking on, as the vendor runs it; system_prompt is a socket.
    assert (upsample["max_length"], upsample["sampling_mode"], upsample["thinking"],
            upsample["use_default_template"], upsample["mtp"]) == (2048, "off", True, True, "auto")
    assert "system_prompt" not in upsample
    assert request["value"].startswith("Aspect ratio: 1:1\nRequest: ")
    assert graph[upsample["clip"][0]]["inputs"]["clip_name"] == "p-h3-image_prompt_upsampler_bf16.safetensors"


@pytest.mark.parametrize("kind", ["int8", "w4a8", "lora"])
def test_each_testing_graph_carries_its_set_with_the_upsampler_off(kind):
    graph = _graph(kind)
    _, unet = _one(graph, "DGXMonarchUNETLoader")
    _, cond = _one(graph, "PH3ImageConditioning")
    _, switch = _one(graph, "ComfySwitchNode")
    assert unet["unet_name"] == FILES[kind][0]
    assert graph[cond["clip"][0]]["inputs"]["clip_name"] == FILES[kind][1]
    assert graph[cond["vae"][0]]["inputs"]["vae_name"] == FILES[kind][2]
    assert switch["switch"] is False
    loras = [node for node in graph.values() if node["class_type"] == "DGXMonarchLoraLoader"]
    assert len(loras) == (1 if kind == "lora" else 0)


def test_the_sweep_reads_an_8_mp_image_at_cfg_1():
    facts = graph_facts(_graph())
    assert facts["klass"] == "image" and facts["megapixels"] == pytest.approx(8.48, abs=0.01)
    assert facts["sampler"] == "euler" and facts["cfg"] is None
    assert set(facts["unets"].values()) == {FILES["bf16"][0]}


def test_the_note_names_every_file_and_the_shift():
    note = next(node["widgets_values"][0] for node in json.loads(PATHS["bf16"].read_text())["nodes"]
                if node["type"] == "Note")
    for name in (*FILES["bf16"], "p-h3-image_prompt_upsampler_bf16.safetensors"):
        assert name in note
    assert "ModelSamplingMiniMaxH3 (DGX Monarch)" in note and "#144" in note
    assert note.isascii()


def test_the_manifest_lists_the_four_files_the_template_loads():
    manifest = (ROOT / "example_workflows" / "artifacts.toml").read_text()
    block = manifest.split(f'"{NAME}" = [', 1)[1].split("]", 1)[0]
    listed = {line.strip().strip('",') for line in block.splitlines() if line.strip()}
    assert listed == {"diffusion_models/" + FILES["bf16"][0], "text_encoders/" + FILES["bf16"][1],
                      "text_encoders/p-h3-image_prompt_upsampler_bf16.safetensors",
                      "vae/" + FILES["bf16"][2]}
    assert "PrunaAI/P-H3-Image-Comfy/resolve/44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543" in manifest


def test_the_lora_route_runs_at_comfys_strength_in_the_graph_and_the_sweep():
    """ComfyUI's own p-h3-image template loads the LoRA at 1.0, so this cell must too."""
    graph = _graph("lora")
    (lora_id, lora), = [(key, node["inputs"]) for key, node in graph.items()
                        if node["class_type"] == "DGXMonarchLoraLoader"]
    assert lora["strength_model"] == 1.0
    name = "dgx-monarch-test-p-h3-image-lora"
    assert name in GRAPH_LORA_STRENGTH_TEMPLATES
    facts = graph_facts(graph)
    assert lora_stack(name, facts, ["p-h3-image_lora_bf16.safetensors"]) == [
        {"node": lora_id, "name": "p-h3-image_lora_bf16.safetensors", "strength": 1.0}]
    # Every other template keeps the placeholder strength, whatever its loader holds.
    assert lora_stack("dgx-monarch-test-minimax-h3-lora", facts, ["any.safetensors"]) == [
        {"node": lora_id, "name": "any.safetensors", "strength": LORA_CELL_STRENGTH}]
    assert lora_stack(name, facts, []) == []
