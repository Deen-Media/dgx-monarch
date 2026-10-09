"""Mage workflows preserve the native conditioning/latent/encoder contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.sweep import convert

from dgx_monarch.nodes.render_validation import _literal_batches

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("turbo", [False, True])
def test_native_mage_conditioning_feeds_both_branches_and_latent(edit, turbo):
    stem = "dgx-monarch-mage-flow-" + ("edit" if edit else "t2i")
    stem += "-turbo" if turbo else ""
    ui = json.loads((ROOT / "example_workflows" / (stem + ".json")).read_text())
    graph = convert.convert(ui, convert.repo_widget_names(ROOT))
    nodes = {n["class_type"]: (key, n["inputs"]) for key, n in graph.items()}
    cond_id, cond = nodes["TextEncodeMageFlowEdit"]
    _, sampler = nodes["DGXMonarchKSampler"]
    assert sampler["positive"] == [cond_id, 0]
    assert sampler["negative"] == [cond_id, 1]
    assert sampler["latent_image"] == [cond_id, 2]
    assert (sampler["steps"], sampler["cfg"]) == ((4, 1.0) if turbo else (30, 5.0))
    assert sampler["sampler_name"] == "euler" and sampler["scheduler"] == "simple"
    assert nodes["CLIPLoader"][1]["type"] == "mage"
    assert nodes["VAELoader"][1]["vae_name"] == "mage_flow_vae_bf16.safetensors"
    assert not any("ModelSampling" in name for name in nodes)
    assert cond["width"] == cond["height"] == (0 if edit else 1024)
    assert ("images.image_1" in cond) is edit
    if edit:
        assert cond["images.image_1"] == [nodes["LoadImage"][0], 0]
    assert _literal_batches(graph) == [1]
    graph[cond_id]["inputs"]["batch_size"] = 2
    assert _literal_batches(graph) == [2]
