"""The MiniMax-H3-Image templates keep upstream's recipe, and their Notes match the code.

ComfyUI's own templates for this family (Comfy-Org/workflow_templates 0.11.105,
image_minimax_h3_image_t2i and image_minimax_h3_image_edit) load the INT8
ConvRot DiT and text encoder, the FP16 VAE and the W4A8 prompt expander, and
sample with KSampler at 6 steps, CFG 1, euler and simple on a zeroed negative,
with the expander behind an If/Else switch that is off in text to image and on
in the edit. The shipped graphs must keep each of those, with the DGX Monarch
loader and sampler in place of the stock ones. The testing graph walks the
checkpoint's own sigma grid through Sampler Custom with the expander off.
What the graphs change on purpose is pinned too: literal canvas numbers, one
stock scale node, and this project's own short instruction decoded greedily.
CPU only: the graphs are read as JSON and no ComfyUI is imported.
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
from dgx_monarch.nodes import gate_proof_request, loader_graph
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
    scale_id, scale = _one(graph, "ImageScaleToTotalPixels")
    size_id, size = _one(graph, "GetImageSize")
    _, expand = _one(graph, "TextGenerateV2")
    assert image["image"] == "reference.png"
    assert scale == {"image": [image_id, 0], "upscale_method": "area", "megapixels": 4.0,
                     "resolution_steps": 32}
    assert size["image"] == [scale_id, 0]
    # image_1 is the picture to edit; both sides come from the scaled picture.
    assert cond["images.image_1"] == [scale_id, 0]
    assert (cond["width"], cond["height"]) == ([size_id, 0], [size_id, 1])
    assert [key for key in cond if key.startswith("images.")] == ["images.image_1"]
    # The expander reads the same picture, under Generate Text's own slot name.
    assert expand["images.image1"] == [scale_id, 0]
    del cond_id


@pytest.mark.parametrize(("name", "switched_on"), [(T2I, False), (EDIT, True), (GRID, False)])
def test_the_expander_sits_behind_the_switch_with_upstreams_default(name, switched_on):
    graph = _graph(name)
    _, cond = _one(graph, "MiniMaxH3ImageConditioning")
    _, preview = _one(graph, "PreviewAny")
    switch_id, switch = _one(graph, "ComfySwitchNode")
    expand_id, expand = _one(graph, "TextGenerateV2")
    texts = {key: node["inputs"]["value"] for key, node in graph.items()
             if node["class_type"] == "PrimitiveStringMultiline"}
    assert switch["switch"] is switched_on
    assert cond["prompt"] == [switch_id, 0] and preview["source"] == [switch_id, 0]
    assert switch["on_true"] == [expand_id, 0]
    request_id = switch["on_false"][0]
    assert expand["prompt"] == [request_id, 0]
    instruction_id = expand["system_prompt"][0]
    assert {request_id, instruction_id} == set(texts)
    assert graph[expand["clip"][0]]["inputs"] == {"clip_name": EXPANDER, "type": "minimax",
                                                  "device": "default"}
    # Greedy decoding with thinking on: a dynamic combo key that reveals no child widget.
    assert (expand["max_length"], expand["sampling_mode"], expand["thinking"],
            expand["use_default_template"], expand["mtp"]) == (8192, "off", True, True, "auto")
    edit = name == EDIT
    assert texts[request_id] == (gen_templates.H3I_EDIT_REQUEST if edit
                                 else gen_templates.H3I_REQUEST)
    assert texts[instruction_id] == (gen_templates.H3I_EDIT_INSTRUCTION if edit
                                     else gen_templates.H3I_INSTRUCTION)
    assert ("images.image1" in expand) is edit


def test_the_texts_are_this_projects_own_and_safe_in_every_box():
    for text in (gen_templates.H3I_REQUEST, gen_templates.H3I_EDIT_REQUEST,
                 gen_templates.H3I_INSTRUCTION, gen_templates.H3I_EDIT_INSTRUCTION):
        assert text.isascii()
        # Nothing the browser's dynamic prompt pass would rewrite, wherever the text lands.
        assert not set("{}|") & set(text) and "//" not in text and "/*" not in text
        assert len(text) < 1000, "a short text of our own, not the vendor's task prompt"
    # The conditioning node presents pictures as Picture 1, Picture 2 (picture_label).
    assert "<Picture 1>" in gen_templates.H3I_EDIT_REQUEST
    assert "<Picture 1>" in gen_templates.H3I_EDIT_INSTRUCTION


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
    assert "this project's own short instruction" in note
    # Nothing here has rendered on hardware, so no Note may claim a result.
    for claim in ("bit-identical", "pixel-identical", "passed", "NRMS"):
        assert claim not in note


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


def test_the_driver_charges_the_expander_only_where_the_switch_is_on(staged):
    """The If/Else switch is lazy, so text to image never loads the expander; the edit does."""
    ratio = loader_graph.TE_RESIDENT_RATIO
    encoder = staged[("text_encoders", ENCODER)]
    expander = staged[("text_encoders", EXPANDER)]
    off = loader_graph.driver_stack_terms(_graph(T2I), family=FAMILY, sp_degree=2)
    assert off.te_artifacts == (ENCODER,)
    assert off.te_bytes == int(encoder * ratio)
    on = loader_graph.driver_stack_terms(_graph(EDIT), family=FAMILY, sp_degree=2)
    assert set(on.te_artifacts) == {ENCODER, EXPANDER}
    # The edit charges the expander on top, at the same resident ratio.
    assert on.te_bytes - off.te_bytes == pytest.approx(expander * ratio, abs=2)
    assert off.vae_artifacts == on.vae_artifacts == (VAE,)
    grid = loader_graph.driver_stack_terms(_graph(GRID), family=FAMILY, sp_degree=2)
    assert grid.te_artifacts == (ENCODER,)


def test_the_loader_prices_a_4_mp_target_for_both_shipped_graphs():
    """A literal 2048 square, and the wired edit canvas at the node's 2048 default."""
    for name in (T2I, EDIT):
        rows = h3_rows.rows_from_prompt_graph(_graph(name), image=True)
        assert rows.resolved and (rows.video, rows.audio, rows.latent_t) == (4096, 0, 1), name
