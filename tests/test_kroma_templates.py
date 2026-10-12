"""Kroma recipes preserve the vendor's native conditioning and two-pass flow."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.sweep import convert

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("dgx-monarch-kroma-t2i", "dgx-monarch-kroma-t2i-int8",
         "dgx-monarch-kroma-t2i-upscale")


def _graph(name):
    ui = json.loads((ROOT / "example_workflows" / f"{name}.json").read_text())
    return convert.convert(ui, convert.repo_widget_names(ROOT))


def _one(graph, kind):
    found = [(key, node["inputs"]) for key, node in graph.items()
             if node["class_type"] == kind]
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("name", NAMES)
def test_kroma_sampling_keeps_native_conditioning_and_shift(name):
    graph = _graph(name)
    init_id, init = _one(graph, "DGXMonarchInit")
    unet_id, unet = _one(graph, "DGXMonarchUNETLoader")
    clip_id, clip = _one(graph, "CLIPLoader")
    pos_id, positive = _one(graph, "CLIPTextEncode")
    neg_id, negative = _one(graph, "ConditioningZeroOut")
    latent_id, latent = _one(graph, "EmptyLatentImage")
    _, vae = _one(graph, "VAELoader")
    assert init["topology"] == init["mode"] == "auto"
    assert init["auto_gate"] == "first_use"
    assert unet["mesh"] == [init_id, 0] and unet["weight_dtype"] == "default"
    expected = ("kromaKrea2_opd.safetensors" if name.endswith("-int8")
                else "kroma-v0.3.1-turbo-opd.safetensors")
    assert unet["unet_name"] == expected
    assert (clip["clip_name"], clip["type"]) == ("qwen3vl_4b_bf16.safetensors", "krea2")
    assert positive["clip"] == [clip_id, 0]
    assert negative["conditioning"] == [pos_id, 0]
    assert (latent["width"], latent["height"], latent["batch_size"]) == (1024, 1024, 1)
    assert vae["vae_name"] == "qwen_image_vae.safetensors"
    assert not any("ModelSampling" in node["class_type"] for node in graph.values())
    samplers = [node["inputs"] for node in graph.values()
                if node["class_type"] == "DGXMonarchKSampler"]
    initial = [node for node in samplers if node["latent_image"] == [latent_id, 0]]
    assert len(initial) == 1
    assert (initial[0]["steps"], initial[0]["denoise"]) == (8, 1.0)
    for sampler in samplers:
        assert sampler["model"] == [unet_id, 0]
        assert sampler["positive"] == [pos_id, 0]
        assert sampler["negative"] == [neg_id, 0]
        assert (sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (
            1.0, "euler", "simple")
    if not name.endswith("-upscale"):
        assert len(samplers) == 1
        decode_id, decode = _one(graph, "VAEDecode")
        sampler_id, _ = _one(graph, "DGXMonarchKSampler")
        _, save = _one(graph, "SaveImage")
        assert decode["samples"] == [sampler_id, 0]
        assert save["images"] == [decode_id, 0]


def test_kroma_refinement_runs_after_pixel_upscale_and_vae_encode():
    graph = _graph("dgx-monarch-kroma-t2i-upscale")
    scale_id, scale = _one(graph, "ImageScaleBy")
    encode_id, encode = _one(graph, "VAEEncode")
    vae_id, _ = _one(graph, "VAELoader")
    _, save = _one(graph, "SaveImage")
    assert (scale["upscale_method"], scale["scale_by"]) == ("lanczos", 2.0)
    first_decode = graph[scale["image"][0]]
    assert first_decode["class_type"] == "VAEDecode"
    first_sample = graph[first_decode["inputs"]["samples"][0]]
    assert first_sample["class_type"] == "DGXMonarchKSampler"
    assert first_sample["inputs"]["denoise"] == 1.0
    assert encode == {"pixels": [scale_id, 0], "vae": [vae_id, 0]}
    final_decode = graph[save["images"][0]]
    assert final_decode["class_type"] == "VAEDecode"
    assert final_decode["inputs"]["vae"] == first_decode["inputs"]["vae"] == [vae_id, 0]
    refine = graph[final_decode["inputs"]["samples"][0]]
    assert refine["class_type"] == "DGXMonarchKSampler"
    assert refine["inputs"]["latent_image"] == [encode_id, 0]
    assert (refine["inputs"]["steps"], refine["inputs"]["denoise"]) == (4, 0.2)
    assert len([node for node in graph.values()
                if node["class_type"] == "DGXMonarchKSampler"]) == 2
