"""The FLUX.3 Image Klein templates hold the vendor's recipe.

The shipped graphs carry the vendor-documented settings: the step-distilled
model at 16 Euler/simple steps and CFG 1, the base model at 25 steps and CFG
3.5, 1024 square, the Qwen3-VL encoder under CLIPLoader type `flux3` and the
pinned Flux 2 VAE. The edit graphs spell ReferenceMultiLatent's resize mode out
in stock nodes (tools/gen_templates.py says why): every reference is scaled to
1 MP, encoded once and appended to both prompts in order, and the sampling
latent is the first reference times zero. The negative is the vendor's per
template: an unused empty prompt at CFG 1, an empty prompt at CFG 3.5, Zero Out
on the distilled edit's positive, and an empty prompt carrying the references
on the base edit. CPU only: the graphs are read as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "example_workflows"
TESTING = REPO / "tests" / "fixtures" / "workflows" / "generated"

DISTILLED = "flux3_image_klein_sd_bf16.safetensors"
BASE = "flux3_image_klein_bf16.safetensors"
ENCODER = "qwen3vl_4b_bf16.safetensors"
VAE = "flux2-vae.safetensors"

# name: (DiT, steps, cfg, edit)
RECIPES = {
    "dgx-monarch-flux3-klein-t2i": (DISTILLED, 16, 1.0, False),
    "dgx-monarch-flux3-klein-base-t2i": (BASE, 25, 3.5, False),
    "dgx-monarch-flux3-klein-edit": (DISTILLED, 16, 1.0, True),
    "dgx-monarch-flux3-klein-base-edit": (BASE, 25, 3.5, True),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class _Graph:
    """Nodes by type and the source of every linked input."""

    def __init__(self, doc: dict):
        self.nodes = {node["id"]: node for node in doc["nodes"]}
        self.links = {link[0]: link for link in doc["links"]}

    def of(self, node_type: str) -> list[dict]:
        return [node for node in self.nodes.values() if node["type"] == node_type]

    def one(self, node_type: str) -> dict:
        found = self.of(node_type)
        assert len(found) == 1, f"expected one {node_type}, found {len(found)}"
        return found[0]

    def source(self, node: dict, name: str) -> tuple[dict, int]:
        slot = next(item for item in node["inputs"] if item["name"] == name)
        assert slot["link"] is not None, f"{node['type']}.{name} is not wired"
        link = self.links[slot["link"]]
        return self.nodes[link[1]], link[2]


def _note(doc: dict) -> str:
    notes = [node["widgets_values"][0] for node in doc["nodes"] if node["type"] == "Note"]
    assert len(notes) == 1
    return notes[0]


def _reference_chain(graph: _Graph, conditioning: tuple[dict, int]) -> list[dict]:
    """The Set Reference Latent nodes a conditioning passes through, first to last."""
    chain = []
    node, _slot = conditioning
    while node["type"] == "ReferenceLatent":
        chain.append(node)
        node, _slot = graph.source(node, "conditioning")
    return list(reversed(chain))


def _encoded_image(graph: _Graph, reference_latent: dict) -> dict:
    """The Load Image node behind one Set Reference Latent, checking the chain."""
    encode, slot = graph.source(reference_latent, "latent")
    assert (encode["type"], slot) == ("VAEEncode", 0)
    vae, _ = graph.source(encode, "vae")
    assert vae["type"] == "VAELoader"
    scale, slot = graph.source(encode, "pixels")
    assert (scale["type"], slot) == ("ImageScaleToTotalPixels", 0)
    # lanczos, 1.0 MP and step 1 round each side as ReferenceMultiLatent's
    # resize mode does at its vendor setting.
    assert scale["widgets_values"] == ["lanczos", 1.0, 1]
    image, slot = graph.source(scale, "image")
    assert (image["type"], slot) == ("LoadImage", 0)
    return image


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_the_shipped_graph_holds_the_vendor_recipe(name):
    dit, steps, cfg, edit = RECIPES[name]
    doc = _load(SHIPPED / f"{name}.json")
    graph = _Graph(doc)

    init = graph.one("DGXMonarchInit")["dgxm_widgets"]
    assert (init["topology"], init["mode"], init["auto_gate"]) == ("auto", "auto", "first_use")
    assert graph.one("DGXMonarchUNETLoader")["dgxm_widgets"]["unet_name"] == dit
    assert graph.one("CLIPLoader")["widgets_values"] == [ENCODER, "flux3", "default"]
    assert graph.one("VAELoader")["widgets_values"] == [VAE]

    sampler = graph.one("DGXMonarchKSampler")
    widgets = sampler["dgxm_widgets"]
    assert (widgets["steps"], widgets["cfg"]) == (steps, cfg)
    assert (widgets["sampler_name"], widgets["scheduler"], widgets["denoise"]) == (
        "euler", "simple", 1.0)
    # The distilled model was trained for 4 to 16 steps, and every count here
    # divides the 10000-row sigma table, so simple lands on the vendor schedule.
    assert 10000 % steps == 0
    assert widgets["seed"] == 42

    assert not graph.of("ReferenceMultiLatent")
    assert not graph.of("ModelSamplingFlux"), "ComfyUI's native shift is the vendor's"
    note = _note(doc)
    for required in (dit, ENCODER, VAE, "FLUX Kommunity License v1.0",
                     "0.41.0 or newer", "Sol-Attn waiver"):
        assert required in note, f"{name}: the Note does not name {required!r}"

    latent, slot = graph.source(sampler, "latent_image")
    positive, _ = graph.source(sampler, "positive")
    negative, _ = graph.source(sampler, "negative")
    if not edit:
        assert latent["type"] == "EmptyFlux2LatentImage" and slot == 0
        assert latent["widgets_values"] == [1024, 1024, 1]
        assert positive["type"] == "CLIPTextEncode"
        assert negative["type"] == "CLIPTextEncode"
        assert negative["widgets_values"] == [""]
        assert not graph.of("LoadImage")
        return

    pos_chain = _reference_chain(graph, graph.source(sampler, "positive"))
    neg_chain = _reference_chain(graph, graph.source(sampler, "negative"))
    assert len(pos_chain) == len(neg_chain) == 1
    for chain in (pos_chain, neg_chain):
        image = _encoded_image(graph, chain[0])
        assert image["widgets_values"] == ["reference.png", "image"]
    text, _ = graph.source(pos_chain[0], "conditioning")
    assert text["type"] == "CLIPTextEncode"
    assert "<ref_image_0>" in text["widgets_values"][0]
    plain_negative, _ = graph.source(neg_chain[0], "conditioning")
    if cfg == 1.0:
        # The vendor order: zero the encoded positive, then add the reference.
        assert plain_negative["type"] == "ConditioningZeroOut"
        assert graph.source(plain_negative, "conditioning")[0] is text
    else:
        # Both CFG samples carry the references, as the reference code packs them.
        assert plain_negative["type"] == "CLIPTextEncode"
        assert plain_negative["widgets_values"] == [""]

    # The sampling latent is the first reference times zero, as
    # ReferenceMultiLatent's empty_latent output is.
    assert latent["type"] == "LatentMultiply" and slot == 0
    assert latent["widgets_values"] == [0.0]
    first_encode, _ = graph.source(pos_chain[0], "latent")
    assert graph.source(latent, "samples") == (first_encode, 0)
    for marker in ("<ref_image_0>", "uly2 or single", "Scale Image to Total Pixels"):
        assert marker in note, f"{name}: the Note does not say {marker!r}"


def test_the_shipped_set_is_two_checkpoints_by_two_tasks():
    shipped = {path.stem for path in SHIPPED.glob("dgx-monarch-flux3-*.json")}
    assert shipped == set(RECIPES)


# name: (base template, DiT, encoder)
QUANT_LEGS = {
    "dgx-monarch-test-flux3-klein-int8": (
        "dgx-monarch-flux3-klein-t2i", "flux3_image_klein_sd_int8_convrot.safetensors",
        "qwen3vl_4b_int8_convrot.safetensors"),
    "dgx-monarch-test-flux3-klein-nvfp4": (
        "dgx-monarch-flux3-klein-t2i", "flux3_image_klein_sd_nvfp4.safetensors", ENCODER),
    "dgx-monarch-test-flux3-klein-base-int8": (
        "dgx-monarch-flux3-klein-base-t2i", "flux3_image_klein_int8_convrot.safetensors",
        ENCODER),
    "dgx-monarch-test-flux3-klein-base-nvfp4": (
        "dgx-monarch-flux3-klein-base-t2i", "flux3_image_klein_nvfp4.safetensors", ENCODER),
}


@pytest.mark.parametrize("name", sorted(QUANT_LEGS))
def test_a_quant_leg_changes_only_its_files(name):
    base, dit, encoder = QUANT_LEGS[name]
    leg = _Graph(_load(TESTING / f"{name}.json"))
    shipped = _Graph(_load(SHIPPED / f"{base}.json"))
    assert leg.one("DGXMonarchUNETLoader")["dgxm_widgets"]["unet_name"] == dit
    assert leg.one("CLIPLoader")["widgets_values"][:2] == [encoder, "flux3"]
    assert (leg.one("DGXMonarchKSampler")["dgxm_widgets"]
            == shipped.one("DGXMonarchKSampler")["dgxm_widgets"])
    assert (leg.one("EmptyFlux2LatentImage")["widgets_values"]
            == shipped.one("EmptyFlux2LatentImage")["widgets_values"])


def test_the_odd_grid_leg_pads_both_phases():
    leg = _Graph(_load(TESTING / "dgx-monarch-test-flux3-klein-odd.json"))
    width, height, batch = leg.one("EmptyFlux2LatentImage")["widgets_values"]
    assert (width, height, batch) == (1040, 1040, 1)
    image_rows = (width // 16) * (height // 16)
    assert image_rows == 4225 and image_rows % 2 == 1
    # The shipped prompt encodes to one 80-row block, so the joint stream is odd too.
    assert (image_rows + 80) % 2 == 1


def test_the_two_reference_leg_chains_both_references_in_order():
    doc = _load(TESTING / "dgx-monarch-test-flux3-klein-edit-2ref.json")
    graph = _Graph(doc)
    sampler = graph.one("DGXMonarchKSampler")
    expected = ["dgxm_test_landscape.png", "dgxm_test_landscape_small.png"]
    for name in ("positive", "negative"):
        chain = _reference_chain(graph, graph.source(sampler, name))
        assert [_encoded_image(graph, node)["widgets_values"][0] for node in chain] == expected
    latent, _ = graph.source(sampler, "latent_image")
    first_encode, _ = graph.source(
        _reference_chain(graph, graph.source(sampler, "positive"))[0], "latent")
    assert latent["type"] == "LatentMultiply"
    assert graph.source(latent, "samples") == (first_encode, 0)
    text = graph.one("CLIPTextEncode")["widgets_values"][0]
    assert "<ref_image_0>" in text and "<ref_image_1>" in text
    assert graph.source(graph.one("ConditioningZeroOut"), "conditioning")[0]["type"] == (
        "CLIPTextEncode")


def test_the_two_reference_sizes_give_odd_73_by_55_grids():
    """ReferenceMultiLatent's resize arithmetic, which the scale node repeats:
    round each side at 1 MP, then the Flux 2 VAE crops to a multiple of 16."""
    def grid(width: int, height: int) -> tuple[int, int]:
        scale = (1.0 * 1024 * 1024 / (width * height)) ** 0.5
        return round(width * scale) // 16, round(height * scale) // 16

    first, second = grid(1600, 1200), grid(1024, 768)
    assert first == second == (73, 55)
    rows = first[0] * first[1]
    assert rows % 2 == 1
    assert rows + 2 * rows + 80 == 12125
    # The sweep's portrait stays even, which is why the leg needs landscapes.
    assert grid(480, 832) == (48, 84)
