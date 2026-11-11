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
prompt boxes are empty and the switch is off in all three graphs: the text
ComfyUI's templates carry there is MiniMax's own, and no file of this
repository holds it. Text to image keeps literal canvas numbers. CPU only: the
graphs are read as JSON and no ComfyUI is imported.
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
    switch_id, switch = _one(graph, "ComfySwitchNode")
    extract_id, extract = _one(graph, "JsonExtractString")
    expand_id, expand = _one(graph, "TextGenerateV2")
    splice_id, splice = _one(graph, "StringReplace")
    assert cond["prompt"] == [switch_id, 0]
    assert switch["on_true"] == [extract_id, 0]
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
            "system_id": system_id, "request_id": request_id}


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


@pytest.mark.parametrize("name", [T2I, EDIT, GRID])
def test_the_two_prompt_boxes_are_empty_and_the_switch_is_off(name):
    """No file of this repository holds MiniMax's task prompt or system prompt."""
    chain = _chain(_graph(name))
    assert chain["task"] == "" and chain["system"] == ""
    assert chain["switch"]["switch"] is False
    # The boxes carry ComfyUI's titles, so a reader can tell the empty two apart.
    titled = {node["title"]: node["widgets_values"] for node in _doc(name)["nodes"]
              if node["type"] == "PrimitiveStringMultiline"}
    assert titled["Text (task_prompt)"] == [""] and titled["Text (system_prompt)"] == [""]
    assert titled["Text (user_prompt)"] == [chain["request"]]


def test_no_text_in_any_generated_graph_is_long_enough_to_be_the_vendors_prompt():
    """ComfyUI's text to image task prompt runs to 15,823 characters, its edit one
    is long too, and its demo caption runs to 3,308. Outside the Note, no widget
    here comes near."""
    for name in PATHS:
        for node in _doc(name)["nodes"]:
            if node["type"] == "Note":
                continue
            for value in node.get("widgets_values") or []:
                assert not isinstance(value, str) or len(value) < 400, (name, node["type"])
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
def test_the_note_says_the_expander_is_wired_but_off_and_where_its_text_lives(name):
    note = _note(name)
    edit = name == EDIT
    assert "The prompt expander of ComfyUI's template is wired but off" in note
    assert "the switch is false and the task prompt and system prompt boxes are empty" in note
    # MiniMax's repository and the two files, by path, and that none of it ships here.
    assert gen_templates.H3I_PROMPT_REPO == "MiniMaxAI/MiniMax-H3-Image-PE"
    assert f"its Hugging Face repository {gen_templates.H3I_PROMPT_REPO}" in note
    task = "prompt/edit/task_prompt.txt" if edit else "prompt/t2i/task_prompt.txt"
    assert f"paste {task} into the task prompt box" in note
    assert "paste prompt/t2i/system_prompt.txt into the system prompt box" in note
    assert "this project does not ship it" in note
    assert "and set the switch to true" in note
    # Where the marker goes in a pasted task prompt, as ComfyUI's own copy has it.
    assert ("User input: {input text}" if edit else "input_text: {input text}") in note
    # With the boxes empty the switch stays off, and what happens if it does not.
    assert "With the task prompt box empty the switch must stay false" in note
    assert f"(docs/TROUBLESHOOTING.md #{expander_prompt.TROUBLESHOOTING})" in note
    # What an expander-on render is worth as a comparison.
    assert "it can change from run to run even at a fixed seed" in note
    assert "compare renders and check identity with the switch false" in note
    # The edit graph differs from ComfyUI's here, and says so; text to image does not.
    assert ("where this graph's defaults differ from ComfyUI's edit template, which ships "
            "the switch true and both boxes filled" in note) is edit
    assert ("ComfyUI's template ships the switch false too" in note) is not edit


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
    """The If/Else switch is lazy: as shipped no graph loads the expander, and
    the same graph with the switch on is charged for it."""
    ratio = loader_graph.TE_RESIDENT_RATIO
    encoder = staged[("text_encoders", ENCODER)]
    expander = staged[("text_encoders", EXPANDER)]
    graph = _graph(name)
    off = loader_graph.driver_stack_terms(graph, family=FAMILY, sp_degree=2)
    assert off.te_artifacts == (ENCODER,)
    assert off.te_bytes == int(encoder * ratio)
    switch_id, _ = _one(graph, "ComfySwitchNode")
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
