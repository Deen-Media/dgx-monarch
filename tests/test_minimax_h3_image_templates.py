"""The MiniMax-H3-Image templates keep upstream's recipe, and their Notes match the code.

ComfyUI's own templates for this family (Comfy-Org/workflow_templates 0.11.105,
image_minimax_h3_image_t2i and image_minimax_h3_image_edit) load the INT8
ConvRot DiT and text encoder, the FP16 VAE and the W4A8 prompt expander, and
sample with KSampler at 6 steps, CFG 1, euler and simple on a zeroed negative,
with the expander behind an If/Else switch. The shipped graphs must keep each
of those, with the DGX Monarch loader and sampler in place of the stock ones,
and the expander chain as ComfyUI wires it: Replace Text on `{input text}`,
Generate Text with ComfyUI's decoding numbers, Extract Text from JSON on
`expanded_caption`. The testing graph walks the checkpoint's own sigma grid
through Sampler Custom. The edit graph keeps upstream's canvas nodes, two
Resize Image/Mask nodes and Get Image Size.

What the graphs change on purpose is pinned too. The task prompt and system
prompt boxes of the two shipped graphs hold this project's own texts, byte
for byte the tracked files under tools/expander_prompts, and the testing graph
keeps both boxes empty: the text ComfyUI's templates carry there is MiniMax's
own, and no file of this repository holds it. The switch follows one value of
the generator, off as shipped, and stays off in the testing graph. An answer
with no caption falls back to the request. Text to image keeps literal canvas
numbers. CPU only: the graphs are read as JSON and no ComfyUI is imported.
"""
from __future__ import annotations

import json
import sys
import tomllib
import types
from pathlib import Path

import pytest
import torch

import minimax_h3_image_headers as h3i
from dgx_monarch import driver_footprint, h3_rows
from dgx_monarch.adapters.minimax_h3 import minimax_h3_topology_would_reject
from dgx_monarch.adapters.minimax_h3_image import FAMILY
from dgx_monarch.nodes import expander_prompt, gate_proof_request, loader_graph
from dgx_monarch.topology import choose_auto_topology, topology_from_preset

ROOT = Path(__file__).resolve().parents[1]
# The benchmark package and the generator live at the checkout root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sweep import convert, matrix  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
try:
    import gen_templates
finally:
    sys.path.remove(str(ROOT / "tools"))

T2I = "dgx-monarch-minimax-h3-image-t2i"
EDIT = "dgx-monarch-minimax-h3-image-edit"
GRID = "dgx-monarch-test-minimax-h3-image-grid"
PATHS = {T2I: ROOT / "example_workflows" / f"{T2I}.json",
         EDIT: ROOT / "example_workflows" / f"{EDIT}.json",
         GRID: ROOT / "tests" / "fixtures" / "workflows" / "generated" / f"{GRID}.json"}
DIT = "minimax_h3_image_pruned_int8_convrot.safetensors"
ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
EXPANDER = "qwen3.5_9b_w4a8_gptq.safetensors"
VAE = "minimax_h3_image_vae_fp16.safetensors"
FILES = {"diffusion_models/" + DIT, "text_encoders/" + ENCODER, "text_encoders/" + EXPANDER,
         "vae/" + VAE}
# The pinned commit of Comfy-Org/MiniMax-H3-Image the four files were checked against.
COMMIT = "892de7cb61df49753bf15145a711154803a355cb"


def _doc(name: str) -> dict:
    return json.loads(PATHS[name].read_text(encoding="utf-8"))


def _graph(name: str) -> dict:
    return convert.convert(_doc(name), convert.repo_widget_names(ROOT))


def _one(graph: dict, kind: str) -> tuple[str, dict]:
    found = [(key, node["inputs"]) for key, node in graph.items() if node["class_type"] == kind]
    assert len(found) == 1, f"expected one {kind}, found {len(found)}"
    return found[0]


def _note(name: str) -> str:
    (note,) = [node["widgets_values"][0] for node in _doc(name)["nodes"] if node["type"] == "Note"]
    return note


def _manifest() -> dict:
    return tomllib.loads((ROOT / "example_workflows" / "artifacts.toml").read_text(encoding="utf-8"))


# ---- The recipe ---------------------------------------------------------------

@pytest.mark.parametrize("name", [T2I, EDIT])
def test_the_shipped_graph_keeps_upstreams_recipe(name):
    graph = _graph(name)
    init_id, init = _one(graph, "DGXMonarchInit")
    unet_id, unet = _one(graph, "DGXMonarchUNETLoader")
    cond_id, cond = _one(graph, "MiniMaxH3ImageConditioning")
    neg_id, neg = _one(graph, "ConditioningZeroOut")
    ks_id, ks = _one(graph, "DGXMonarchKSampler")
    _, decode = _one(graph, "VAEDecode")
    assert (init["topology"], init["mode"], init["auto_gate"]) == ("auto", "auto", "first_use")
    assert (unet["unet_name"], unet["weight_dtype"], unet["mesh"]) == (DIT, "default", [init_id, 0])
    assert graph[cond["clip"][0]]["inputs"] == {"clip_name": ENCODER, "type": "minimax",
                                                "device": "default"}
    assert graph[cond["vae"][0]]["inputs"] == {"vae_name": VAE}
    assert neg["conditioning"] == [cond_id, 0]
    assert (ks["model"], ks["positive"], ks["negative"], ks["latent_image"]) == (
        [unet_id, 0], [cond_id, 0], [neg_id, 0], [cond_id, 1])
    assert (ks["seed"], ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"],
            ks["denoise"]) == (42, 6, 1.0, "euler", "simple", 1.0)
    assert decode["samples"] == [ks_id, 0] and decode["vae"] == cond["vae"]
    # No shift node: the family's sampling is the checkpoint's own grid.
    assert not [node for node in graph.values() if "ModelSampling" in node["class_type"]]
    assert not [node for node in graph.values() if node["class_type"] == "DGXMonarchLoraLoader"]


def test_text_to_image_sets_a_literal_4_mp_canvas_and_wires_no_picture():
    _, cond = _one(_graph(T2I), "MiniMaxH3ImageConditioning")
    # Upstream's Resolution Selector at 1:1, 4 MP and a multiple of 32 gives 2048 square.
    assert (cond["width"], cond["height"]) == (2048, 2048)
    assert not [key for key in cond if key.startswith("images.")]
    assert not [node for node in _graph(T2I).values() if node["class_type"] == "LoadImage"]


def test_the_edit_scales_its_picture_to_4_mp_and_sizes_the_render_from_it():
    graph = _graph(EDIT)
    cond_id, cond = _one(graph, "MiniMaxH3ImageConditioning")
    image_id, image = _one(graph, "LoadImage")
    size_id, size = _one(graph, "GetImageSize")
    _, expand = _one(graph, "TextGenerateV2")
    assert image["image"] == "reference.png"
    # Upstream's two Resize Image/Mask nodes, in its order and with its values:
    # to 4 MP by area, then to a multiple of 32.
    resized = {key: node["inputs"] for key, node in graph.items()
               if node["class_type"] == "ResizeImageMaskNode"}
    (scale_id,) = [key for key, inputs in resized.items() if inputs["input"] == [image_id, 0]]
    (multiple_id,) = [key for key in resized if key != scale_id]
    assert resized[scale_id] == {"input": [image_id, 0], "resize_type": "scale total pixels",
                                 "resize_type.megapixels": 4.0, "scale_method": "area"}
    assert resized[multiple_id] == {"input": [scale_id, 0], "resize_type": "scale to multiple",
                                    "resize_type.multiple": 32, "scale_method": "area"}
    assert not [node for node in graph.values() if node["class_type"] == "ImageScaleToTotalPixels"]
    assert size["image"] == [multiple_id, 0]
    # image_1 is the picture to edit; both sides come from the resized picture.
    assert cond["images.image_1"] == [multiple_id, 0]
    assert (cond["width"], cond["height"]) == ([size_id, 0], [size_id, 1])
    assert [key for key in cond if key.startswith("images.")] == ["images.image_1"]
    # The expander reads the same picture, under Generate Text's own slot name.
    assert expand["images.image1"] == [multiple_id, 0]
    del cond_id


def _chain(graph: dict) -> dict:
    """The expander chain's nodes by role, read back from the conditioning node's prompt."""
    _, cond = _one(graph, "MiniMaxH3ImageConditioning")
    extract_id, extract = _one(graph, "JsonExtractString")
    expand_id, expand = _one(graph, "TextGenerateV2")
    splice_id, splice = _one(graph, "StringReplace")
    # Two If/Else switches: the one a reader sets feeds the conditioning node,
    # and the empty-caption fallback feeds that one's on_true.
    switches = {key: node["inputs"] for key, node in graph.items()
                if node["class_type"] == "ComfySwitchNode"}
    assert len(switches) == 2
    switch_id = cond["prompt"][0]
    switch = switches[switch_id]
    (fallback_id,) = [key for key in switches if key != switch_id]
    fallback = switches[fallback_id]
    assert cond["prompt"] == [switch_id, 0]
    assert switch["on_true"] == [fallback_id, 0]
    assert extract["json_string"] == [expand_id, 0]
    assert expand["prompt"] == [splice_id, 0]
    request_id, task_id, system_id = (
        splice["replace"][0], splice["string"][0], expand["system_prompt"][0])
    assert switch["on_false"] == [request_id, 0]
    texts = {key: node["inputs"]["value"] for key, node in graph.items()
             if node["class_type"] == "PrimitiveStringMultiline"}
    assert set(texts) == {request_id, task_id, system_id} and len(texts) == 3
    return {"cond": cond, "switch": switch, "extract": extract, "expand": expand,
            "splice": splice, "request": texts[request_id], "task": texts[task_id],
            "system": texts[system_id], "switch_id": switch_id, "task_id": task_id,
            "system_id": system_id, "request_id": request_id, "fallback": fallback,
            "fallback_id": fallback_id, "extract_id": extract_id}


@pytest.mark.parametrize("name", [T2I, EDIT, GRID])
def test_the_expander_chain_is_wired_as_comfyui_wires_it(name):
    graph = _graph(name)
    chain = _chain(graph)
    edit = name == EDIT
    # Replace Text puts the request where the task prompt says {input text};
    # ComfyUI's templates store ["", "{input text}", ""] with both ends wired.
    assert chain["splice"]["find"] == gen_templates.H3I_MARKER == "{input text}"
    assert chain["extract"]["key"] == "expanded_caption"
    expand = chain["expand"]
    assert graph[expand["clip"][0]]["inputs"] == {"clip_name": EXPANDER, "type": "minimax",
                                                  "device": "default"}
    # thinking and max_length as upstream stores them: 8192 in its text to
    # image graph, 8196 in its edit graph.
    assert (expand["max_length"], expand["thinking"], expand["use_default_template"],
            expand["mtp"]) == (8196 if edit else 8192, True, True, "auto")
    assert {key: value for key, value in expand.items()
            if key.startswith("sampling_mode")} == gen_templates.H3I_DECODING
    assert ("images.image1" in expand) is edit
    # Preview as Text shows the switch's choice from beside the path.
    _, preview = _one(graph, "PreviewAny")
    assert preview["source"] == [chain["switch_id"], 0]
    assert chain["request"] == (gen_templates.H3I_EDIT_REQUEST if edit
                                else gen_templates.H3I_REQUEST)


def test_generate_text_holds_comfyuis_decoding_numbers():
    """ComfyUI's own two templates store these thirteen values, max_length apart."""
    assert gen_templates.H3I_DECODING == {
        "sampling_mode": "on", "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 20,
        "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.0,
        "sampling_mode.repetition_penalty": 1.0, "sampling_mode.presence_penalty": 0.0,
        "sampling_mode.seed": 0}
    children = gen_templates.DYNAMIC_WIDGETS["TextGenerateV2"]["sampling_mode"]
    assert [f"sampling_mode.{child}" for child in children["on"]] == [
        key for key in gen_templates.H3I_DECODING if key != "sampling_mode"]
    stored = {name: [node["widgets_values"] for node in _doc(name)["nodes"]
                     if node["type"] == "TextGenerateV2"] for name in PATHS}
    assert stored[T2I] == stored[GRID] == [
        ["", 8192, "on", 1, 20, 0.95, 0, 1, 0, 0, True, True, "auto"]]
    assert stored[EDIT] == [["", 8196, "on", 1, 20, 0.95, 0, 1, 0, 0, True, True, "auto"]]


PROMPTS = ROOT / "tools" / "expander_prompts" / "minimax_h3_image"
# The texts as the prompt author's last round left them, by SHA-256.
PROMPT_SHA256 = {
    "system_prompt.txt": "949563c46a6a2322c209159c72fc396f6ed0194ac77b51dcc879ff89ea93298d",
    "t2i_task_prompt.txt": "92df9775562d8731aef197fd00b4a482e2543ebb2507d02ef43b3aad48caf73a",
    "edit_task_prompt.txt": "f61af404739b97da169a89632b654ecbabf4f926ec5aafbff174deef614bd52f",
}


def _prompt_file(name: str) -> str:
    return (PROMPTS / name).read_bytes().decode("utf-8")


@pytest.mark.parametrize("name", [T2I, EDIT])
def test_the_shipped_boxes_hold_this_projects_texts_byte_for_byte(name):
    """Each box is its tracked file, as the UI graph stores it and as the API prompt sends it."""
    task_file = "edit_task_prompt.txt" if name == EDIT else "t2i_task_prompt.txt"
    chain = _chain(_graph(name))
    assert chain["task"] == _prompt_file(task_file)
    assert chain["system"] == _prompt_file("system_prompt.txt")
    titled = {node["title"]: node["widgets_values"] for node in _doc(name)["nodes"]
              if node["type"] == "PrimitiveStringMultiline"}
    assert titled["Text (task_prompt)"] == [_prompt_file(task_file)]
    assert titled["Text (system_prompt)"] == [_prompt_file("system_prompt.txt")]
    assert titled["Text (user_prompt)"] == [chain["request"]]
    # The JSON encoder wrote them: the file's bytes sit in the template as one JSON string.
    raw = PATHS[name].read_text(encoding="utf-8")
    assert json.dumps(_prompt_file(task_file)) in raw
    assert json.dumps(_prompt_file("system_prompt.txt")) in raw


def test_the_tracked_texts_are_the_final_ones_and_fit_the_chain():
    import hashlib

    assert sorted(path.name for path in PROMPTS.iterdir()) == sorted(PROMPT_SHA256)
    for name, digest in PROMPT_SHA256.items():
        data = (PROMPTS / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        text = data.decode("ascii")
        assert "\r" not in text and text.endswith("\n")
        # Replace Text fills the marker once in a task prompt; the system prompt has none.
        assert text.count(gen_templates.H3I_MARKER) == (0 if name == "system_prompt.txt" else 1)
        # The answer's key is the one Extract Text from JSON reads.
        assert '{"expanded_caption": ' in text
    assert set(gen_templates.H3I_PROMPT_FILES.values()) == set(PROMPT_SHA256)
    # The edit text names pictures as the conditioning node presents them.
    assert "<Picture 1>" in _prompt_file("edit_task_prompt.txt")
    assert "<Picture" not in _prompt_file("t2i_task_prompt.txt")


def test_the_testing_graph_keeps_both_boxes_empty_and_its_switch_off():
    """No sweep or identity row of the testing graph can run a sampled language model."""
    chain = _chain(_graph(GRID))
    assert chain["task"] == "" and chain["system"] == ""
    assert chain["switch"]["switch"] is False
    titled = {node["title"]: node["widgets_values"] for node in _doc(GRID)["nodes"]
              if node["type"] == "PrimitiveStringMultiline"}
    assert titled["Text (task_prompt)"] == [""] and titled["Text (system_prompt)"] == [""]
    assert "task prompt and system prompt boxes are empty" in _note(GRID)


@pytest.mark.parametrize("name", [T2I, EDIT])
def test_the_shipped_switch_follows_the_generators_one_value(name):
    """Shipping the expander on is this one value plus regenerated files."""
    chain = _chain(_graph(name))
    assert chain["switch"]["switch"] is gen_templates.H3I_EXPANDER_ON
    titles = {node["title"] for node in _doc(name)["nodes"] if node["type"] == "ComfySwitchNode"}
    assert titles == {gen_templates.H3I_SWITCH_TITLE, gen_templates.H3I_FALLBACK_TITLE}
    # The refusal names the switch to set by the title it carries here.
    for sentence in (expander_prompt.EMPTY_TASK_PROMPT, expander_prompt.MISSING_MARKER):
        assert f"set the switch titled {gen_templates.H3I_SWITCH_TITLE} to false" in sentence
    note = _note(name)
    assert ("As shipped that switch is true" in note) is gen_templates.H3I_EXPANDER_ON
    assert ("As shipped that switch is false" in note) is not gen_templates.H3I_EXPANDER_ON


def test_the_docs_say_what_the_shipped_switch_is():
    """The generator's one value changes the graphs and their Notes. The docs that say
    both templates ship the switch false are prose, and must change with it: this
    test fails until they do, and names them."""
    off = not gen_templates.H3I_EXPANDER_ON
    models = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")
    install = (ROOT / "docs" / "INSTALL.md").read_text(encoding="utf-8")
    assert ("both templates ship the switch false, so as shipped each renders from the\n"
            "request as written" in models) is off
    assert ("- Both templates ship the switch false, where ComfyUI's ship it true" in models) is off
    assert ("Both templates ship it false, so as\nshipped neither loads it" in install) is off


def test_the_source_archive_carries_the_three_texts():
    """The generator reads the texts when it is imported, so a source archive without
    them cannot build a template. MANIFEST.in decides what the archive holds."""
    lines = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "recursive-include tools/expander_prompts *.txt" in lines
    assert all(path.suffix == ".txt" for path in PROMPTS.iterdir())


@pytest.mark.parametrize("name", [T2I, EDIT, GRID])
def test_an_empty_caption_falls_back_to_the_request(name):
    """Extract Text from JSON returns "" for an answer with no caption. Compare Text
    tests the trimmed caption against the empty text and drives the second If/Else,
    which hands on the request when it is empty and the caption as extracted otherwise."""
    graph = _graph(name)
    chain = _chain(graph)
    trim_id, trim = _one(graph, "StringTrim")
    compare_id, compare = _one(graph, "StringCompare")
    assert trim == {"string": [chain["extract_id"], 0], "mode": "Both"}
    assert compare == {"string_a": [trim_id, 0], "string_b": "", "mode": "Equal",
                       "case_sensitive": True}
    assert chain["fallback"] == {"switch": [compare_id, 0],
                                 "on_true": [chain["request_id"], 0],
                                 "on_false": [chain["extract_id"], 0]}
    # Preview as Text reads the outer switch, so it shows the text that is rendered:
    # the request with the switch false or after a fallback, the caption otherwise.
    _, preview = _one(graph, "PreviewAny")
    assert preview["source"] == chain["cond"]["prompt"] == [chain["switch_id"], 0]


def test_no_text_outside_the_note_and_the_two_prompt_boxes_is_long():
    """ComfyUI's text to image task prompt runs to 15,823 characters, its edit one
    is long too, and its demo caption runs to 3,308. This project's own texts run to
    1,076, 2,860 and 3,298 bytes, and no other widget comes near."""
    ours = {_prompt_file(name) for name in PROMPT_SHA256}
    assert max(len(text) for text in ours) < 4000
    for name in PATHS:
        for node in _doc(name)["nodes"]:
            if node["type"] == "Note":
                continue
            for value in node.get("widgets_values") or []:
                assert (not isinstance(value, str) or len(value) < 400
                        or value in ours), (name, node["type"])
    for text in (gen_templates.H3I_REQUEST, gen_templates.H3I_EDIT_REQUEST):
        assert text.isascii() and len(text) < 400
        # Nothing the browser's dynamic prompt pass would rewrite.
        assert not set("{}|") & set(text) and "//" not in text and "/*" not in text
    # The conditioning node presents pictures as Picture 1, Picture 2 (picture_label).
    assert "<Picture 1>" in gen_templates.H3I_EDIT_REQUEST


# ---- The generator's autogrow rule ----------------------------------------------

def test_a_wired_picture_grows_one_socket_and_an_unwired_node_keeps_none():
    by_type = {}
    for name in (T2I, EDIT, "p-h3-image"):
        path = PATHS.get(name, ROOT / "example_workflows" / f"dgx-monarch-{name}.json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        by_type[name] = {node["type"]: [(slot["name"], slot["type"]) for slot in node["inputs"]]
                         for node in doc["nodes"]}
    # ComfyUI's schema order: the pictures follow the declared sockets, ahead of the
    # wired widgets, and Generate Text's forced system prompt comes after its pictures.
    assert by_type[EDIT]["MiniMaxH3ImageConditioning"] == [
        ("clip", "CLIP"), ("vae", "VAE"), ("images.image_1", "IMAGE"), ("prompt", "STRING"),
        ("width", "INT"), ("height", "INT")]
    assert by_type[EDIT]["TextGenerateV2"] == [
        ("clip", "CLIP"), ("images.image1", "IMAGE"), ("system_prompt", "STRING"),
        ("prompt", "STRING")]
    assert by_type[T2I]["MiniMaxH3ImageConditioning"] == [
        ("clip", "CLIP"), ("vae", "VAE"), ("prompt", "STRING")]
    # The rule adds nothing to a node that wires no picture, p-h3-image's included.
    assert by_type[T2I]["TextGenerateV2"] == by_type["p-h3-image"]["TextGenerateV2"] == [
        ("clip", "CLIP"), ("system_prompt", "STRING"), ("prompt", "STRING")]


# ---- The grid testing graph ---------------------------------------------------

def test_the_testing_graph_walks_the_checkpoints_own_grid_through_sampler_custom():
    graph = _graph(GRID)
    unet_id, _ = _one(graph, "DGXMonarchUNETLoader")
    cond_id, _ = _one(graph, "MiniMaxH3ImageConditioning")
    guider_id, guider = _one(graph, "DGXMonarchBasicGuider")
    _, sampler = _one(graph, "KSamplerSelect")
    _, noise = _one(graph, "RandomNoise")
    sigmas_id, sigmas = _one(graph, "ManualSigmas")
    _, custom = _one(graph, "DGXMonarchSamplerCustom")
    assert guider == {"model": [unet_id, 0], "conditioning": [cond_id, 0]}
    assert (sampler["sampler_name"], noise["noise_seed"]) == ("euler", 42)
    assert custom["guider"] == [guider_id, 0] and custom["sigmas"] == [sigmas_id, 0]
    assert custom["latent_image"] == [cond_id, 1]
    assert not [node for node in graph.values() if node["class_type"] == "DGXMonarchKSampler"]
    # The string is the published header's grid, and float32 reads it back exactly.
    assert sigmas["sigmas"] == gen_templates.H3I_GRID == ", ".join(repr(v) for v in h3i.GRID)
    typed = torch.tensor([float(part) for part in sigmas["sigmas"].split(",")])
    assert torch.equal(typed, torch.tensor(h3i.GRID, dtype=torch.float32))
    assert len(typed) == h3i.CONFIG["transformer"]["pdd_steps"] + 1
    # The one-step probe keeps the two ends of the grid.
    assert convert.probe_sigmas(sigmas["sigmas"], 1) == "1.0, 0.0"


def test_each_graphs_first_use_proof_is_the_one_its_note_names():
    """KSampler proves with two KSampler steps; Sampler Custom with its own first two."""
    light, steps = gate_proof_request.proof_request(
        {"kind": "ksampler", "steps": 6, "sampler_name": "euler", "scheduler": "simple"})
    assert (light["kind"], steps) == ("ksampler_advanced", 2)
    for name in (T2I, EDIT):
        assert "each proof render is two KSampler steps, 1.0 to 0.75 to 0.0" in _note(name)
    grid = torch.tensor(h3i.GRID, dtype=torch.float32)
    light, steps = gate_proof_request.proof_request({"kind": "custom", "sigmas": grid})
    assert steps == 2 and torch.equal(light["sigmas"], grid[:3])
    assert "its own first two steps, 1.0 to 0.9675 to 0.8875" in _note(GRID)
    assert [round(float(value), 4) for value in grid[:3]] == [1.0, 0.9675, 0.8875]


# ---- The Notes ----------------------------------------------------------------

@pytest.mark.parametrize("name", [T2I, EDIT])
def test_the_note_names_the_files_the_topologies_the_gate_and_the_record(name):
    note = _note(name)
    assert note.isascii()
    for file in (DIT, ENCODER, EXPANDER, VAE):
        assert file in note
    assert "auto asks for sequence-parallel (uly2) on a configured pair" in note
    assert "a world of one GPU runs stock ComfyUI's forward" in note
    assert gen_templates.GATE in note and gen_templates.DERIVED in note
    assert "docs/VALIDATION.md, under MiniMax-H3-Image" in note and "NOT RUN" in note
    assert "ComfyUI 0.45.0 or newer" in note
    assert "ModelSamplingMiniMaxH3 (DGX Monarch) refuses this family" in note
    # Nothing here has rendered on hardware, so no Note may claim a result.
    for claim in ("bit-identical", "pixel-identical", "passed", "NRMS"):
        assert claim not in note


@pytest.mark.parametrize("name", [T2I, EDIT])
def test_the_note_says_whose_the_texts_are_and_what_the_expander_costs(name):
    note = _note(name)
    edit = name == EDIT
    on = gen_templates.H3I_EXPANDER_ON
    assert ("A prompt expander is wired and off" in note) is not on
    assert f"the switch titled {gen_templates.H3I_SWITCH_TITLE} picks that caption or the request" in note
    # The texts are this project's own, and nothing of MiniMax's is here.
    assert ("The task prompt and system prompt boxes hold this project's own texts, "
            "written from public research on the model. They are not MiniMax's prompts") in note
    assert gen_templates.H3I_PROMPT_REPO == "MiniMaxAI/MiniMax-H3-Image-PE"
    assert (f"which MiniMax publishes in its Hugging Face repository "
            f"{gen_templates.H3I_PROMPT_REPO} under its own licence; nothing of those is in "
            "this graph") in note
    # No sentence tells a user to paste the vendor's files.
    assert "paste" not in note and "prompt/t2i" not in note and "prompt/edit" not in note
    # What turning it on costs: the expander's memory and the tokens it may write.
    tokens = 8196 if edit else 8192
    assert (f"the driver loads the 7.3 GB expander beside the text encoder, and Generate "
            f"Text may write up to {tokens} tokens before the render starts") in note
    assert ("a wait no hardware run has timed yet" in note) is not on
    # A user may replace the texts, and what a replacement must keep.
    assert ("You may replace either text with your own: keep {input text} in the task "
            "prompt where the request belongs, and have the answer hold the caption under "
            "the key expanded_caption") in note
    # The fallback, and that the preview shows what is rendered.
    assert (f"If the answer holds no caption, the switch titled "
            f"{gen_templates.H3I_FALLBACK_TITLE} sends the request as written instead, and "
            "Preview as Text shows the text that is rendered either way") in note
    # The refusal, and what an expander-on render is worth as a comparison.
    assert ("refuses a graph whose expander is on with an empty task prompt, or with one "
            "that lacks the marker "
            f"(docs/TROUBLESHOOTING.md #{expander_prompt.TROUBLESHOOTING})") in note
    assert "it can change from run to run even at a fixed seed" in note
    assert "compare renders and check identity with the switch false" in note
    # Both of ComfyUI's templates ship the switch true, with MiniMax's texts.
    assert ("ComfyUI's " + ("edit" if edit else "text to image")
            + " template ships its switch true, through "
            + ("a Boolean box" if edit else "its subgraph's enable_PE input")
            + ", with MiniMax's texts in the two boxes") in note


def test_the_notes_topology_sentences_match_the_auto_table_and_the_driver():
    megapixels = matrix.spatial_megapixels(2048, 2048)
    for quant in ("bf16", "int8", "w4a8"):
        assert choose_auto_topology(FAMILY, quant, megapixels, 2,
                                    cfg_value=1.0).topology.describe() == "uly2"
        assert choose_auto_topology(FAMILY, quant, megapixels, 1,
                                    cfg_value=1.0).topology.describe() == "single"
    for preset in ("cfg2", "dp2"):
        topo = topology_from_preset(preset, 2)
        assert minimax_h3_topology_would_reject(topo.cfg, topo.dp)
    assert "cfg2 and dp2 refuse" in _note(T2I)


def test_the_notes_sizes_are_the_manifests_bytes():
    rows = {row["file"]: row for row in _manifest()["artifact"]}
    said = {DIT: "21.0 GB DiT", ENCODER: "27.1 GB text encoder", EXPANDER: "7.3 GB expander",
            VAE: "1.2 GB VAE"}
    for file, phrase in said.items():
        assert phrase.startswith(f"{rows[file]['size'] / 1e9:.1f} GB"), file
        for name in (T2I, EDIT):
            assert phrase in _note(name)


def test_the_testing_note_says_what_the_graph_is_for():
    note = _note(GRID)
    assert note.startswith("Testing-only workflow for the sweep harness")
    assert f"Based on {T2I}" in note and "The expander is off" in note
    assert note.isascii()


# ---- The manifest -------------------------------------------------------------

def test_the_manifest_lists_the_four_files_each_graph_loads():
    manifest = _manifest()
    lists = manifest["templates"]
    assert set(lists[T2I]) == set(lists[EDIT]) == FILES
    assert set(lists[f"tests/fixtures/workflows/generated/{GRID}"]) == FILES
    rows = {f"{row['dest']}/{row['file']}": row for row in manifest["artifact"]}
    for entry in FILES:
        row = rows[entry]
        folder = entry.split("/", 1)[0]
        assert row["url"] == (f"https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/{COMMIT}/"
                              f"{folder}/{row['file']}")
        assert row["size"] > 0 and len(row["sha256"]) == 64


# ---- The sweep ----------------------------------------------------------------

def test_the_sweep_reads_the_canvas_the_sampler_and_a_one_step_probe():
    facts = matrix.graph_facts(_graph(T2I))
    assert facts["klass"] == "image" and facts["megapixels"] == pytest.approx(4.19, abs=0.01)
    assert (facts["sampler"], facts["cfg"], facts["probe"]) == ("euler", 1.0, True)
    assert set(facts["unets"].values()) == {DIT} and not facts["packed"]
    # An edit sizes itself from its picture, so the graph holds no canvas to read.
    edit = matrix.graph_facts(_graph(EDIT))
    assert edit["megapixels"] == 0 and edit["probe"] is True
    grid = matrix.graph_facts(_graph(GRID))
    assert (grid["cfg"], grid["sampler"], grid["probe"]) == (None, "euler", True)


def test_the_three_published_dits_share_one_stem():
    stems = {matrix.split_quant(f"minimax_h3_image_pruned_{kind}")[0] for kind in h3i.KINDS}
    assert stems == {"minimaxh3imagepruned"}
    assert {matrix.split_quant(f"minimax_h3_image_pruned_{kind}")[1] for kind in h3i.KINDS} == {
        "bf16", "int8_convrot", "w4a8"}


def _sweep_cells(tmp_path: Path) -> list[dict]:
    """The cells the matrix derives for the three graphs over the three published headers."""
    models = tmp_path / "comfy" / "models" / "diffusion_models"
    graphs = tmp_path / "out" / "graphs"
    models.mkdir(parents=True)
    graphs.mkdir(parents=True)
    for kind in h3i.KINDS:
        h3i.write(models / f"minimax_h3_image_pruned_{kind}.safetensors", kind, blocks=2)
    for name in PATHS:
        (graphs / f"{name}.json").write_text(json.dumps(_graph(name)))
    config = dict(matrix.DEFAULTS)
    config.update(out_dir=tmp_path / "out", comfy_dir=tmp_path / "comfy", model_dir=models,
                  template_dirs=[], exclude=[], toggles={}, toggle_presets={},
                  toggle_combos=[], loras={}, lora_off={}, artifact_groups={},
                  image_presets=["auto", "uly2", "cfg2", "dp2", "cfg2+fsdp", "uly2+fsdp"],
                  video_presets=[], image_attention=["TORCH_FLASH"],
                  video_attention=["TORCH_FLASH"], sol_families=[])
    return matrix.build_cells(config)[0]


def test_the_sweep_labels_the_family_as_the_runtime_answers(tmp_path):
    cells = [cell for cell in _sweep_cells(tmp_path) if cell["session"] == "cluster"]
    assert {cell["family"] for cell in cells} == {FAMILY}
    assert {cell["template"] for cell in cells} == set(PATHS)
    assert {cell["artifact_quant"] for cell in cells} == {"bf16", "int8_convrot", "w4a8"}
    for cell in cells:
        where = (cell["template"], cell["artifact_quant"], cell["preset"])
        if cell["preset"] in ("cfg2", "dp2", "cfg2+fsdp"):
            # The driver's cfg and dp preflight, ahead of the pin guard every file would meet.
            assert cell["label"] == "refuse:P", where
        elif cell["preset"] in ("auto", "uly2"):
            assert cell["label"] == "render", where
        elif cell["artifact_quant"] == "w4a8":
            assert cell["label"] == "refuse:P", where  # FSDP refuses the W4A8 layout


def test_a_world_of_one_gpu_renders_every_published_file_as_the_notes_say(tmp_path):
    """Every published DiT pins INT8 attention, and a world of one GPU keeps the pins
    as stock does, with no waiver card: each local cell of the three graphs renders."""
    local = [cell for cell in _sweep_cells(tmp_path) if cell["session"] == "local"]
    assert {cell["template"] for cell in local} == set(PATHS)
    assert {cell["artifact_quant"] for cell in local} == {"bf16", "int8_convrot", "w4a8"}
    assert {(cell["preset"], cell["label"]) for cell in local} == {("single", "render")}
    assert "a world of one GPU keeps the pins, as stock does" in _note(T2I)


# ---- The driver's price behind the switch --------------------------------------

@pytest.fixture
def staged(monkeypatch, tmp_path):
    """The four files as the manifest sizes them, behind a stub folder_paths."""
    rows = {(row["dest"], row["file"]): row["size"] for row in _manifest()["artifact"]}
    sizes: dict[str, int] = {}
    for entry in FILES:
        folder, name = entry.split("/", 1)
        target = tmp_path / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\0")
        sizes[str(target)] = rows[(folder, name)]
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda path: sizes.get(str(path), 0))
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        str(tmp_path / folder / name) if f"{folder}/{name}" in FILES else None)
    fp.get_filename_list = lambda folder: sorted(  # type: ignore[attr-defined]
        entry.split("/", 1)[1] for entry in FILES if entry.startswith(folder + "/"))
    fp.get_annotated_filepath = lambda name: str(tmp_path / "input" / name)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    return rows


@pytest.mark.parametrize("name", [T2I, EDIT, GRID])
def test_the_driver_charges_the_expander_only_when_the_switch_is_on(staged, name):
    """The outer If/Else switch is lazy: with it false no graph loads the expander,
    and the same graph with it true is charged for the expander, whichever way the
    empty-caption fallback would go."""
    ratio = loader_graph.TE_RESIDENT_RATIO
    encoder = staged[("text_encoders", ENCODER)]
    expander = staged[("text_encoders", EXPANDER)]
    graph = _graph(name)
    switch_id = _chain(graph)["switch_id"]
    graph[switch_id]["inputs"]["switch"] = False
    off = loader_graph.driver_stack_terms(graph, family=FAMILY, sp_degree=2)
    assert off.te_artifacts == (ENCODER,)
    assert off.te_bytes == int(encoder * ratio)
    graph[switch_id]["inputs"]["switch"] = True
    on = loader_graph.driver_stack_terms(graph, family=FAMILY, sp_degree=2)
    assert set(on.te_artifacts) == {ENCODER, EXPANDER}
    # The expander comes on top, at the same resident ratio.
    assert on.te_bytes - off.te_bytes == pytest.approx(expander * ratio, abs=2)
    assert off.vae_artifacts == on.vae_artifacts == (VAE,)


def test_the_loader_prices_a_4_mp_target_for_both_shipped_graphs():
    """A literal 2048 square, and the wired edit canvas at the node's 2048 default."""
    for name in (T2I, EDIT):
        rows = h3_rows.rows_from_prompt_graph(_graph(name), image=True)
        assert rows.resolved and (rows.video, rows.audio, rows.latent_t) == (4096, 0, 1), name
