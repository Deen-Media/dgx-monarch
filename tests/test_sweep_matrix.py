"""The sweep matrix labels a cell the way the runtime would refuse it.

Tiny checkpoints with real signature keys drive the same header sniff the
driver uses, so these labels come from the shipped guards rather than from a
table in the test.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.sweep import compare, convert, driver, matrix, run  # noqa: E402

FAMILY_KEYS = {
    "boogu": ("double_stream_layers.0.weight", "img_instruct_attn.a.weight"),
    "krea2": ("txtfusion.a.weight",),
    "omnigen2": ("time_caption_embed.timestep_embedder.a.weight", "layers.0.weight"),
    "pixeldit": ("pixel_embedder.a.weight", "pixel_blocks.0.weight"),
    "minimax_h3": ("video_patch_proj.weight", "audio_patch_proj.weight"),
    "ideogram4": ("llm_cond_proj.weight", "pixel_blocks.0.weight"),
    "chroma": ("distilled_guidance_layer.a.weight",),
    "flux": ("double_blocks.0.img_attn.qkv.weight", "single_blocks.0.weight",
             "vector_in.a.weight"),
    "flux2": ("double_stream_modulation_img.0.weight",),
    "anima": ("llm_adapter.a.weight", "mlp.layer1.weight"),
    "ernie": ("adaLN_sa_ln.a.weight", "mlp.linear_fc2.weight"),
    "orphan": ("nothing.weight",),
    "zimage": ("cap_embedder.0.weight", "x_pad_token"),
}
PRESETS = ["auto", "uly2", "cfg2", "ring2", "uly2+fsdp", "cfg2+fsdp", "ring2+fsdp"]


def _checkpoint(directory: Path, name: str, family: str, dtype=torch.bfloat16) -> None:
    save_file({key: torch.zeros(2, 2, dtype=dtype) for key in FAMILY_KEYS[family]},
              str(directory / name))


def _graph(unet: str, video: bool = False, cfg: float = 3.5,
           sampler: str = "euler") -> dict:
    latent = {"width": 1024, "height": 1024, "batch_size": 1}
    if video:
        latent["length"] = 49
    return {
        "1": {"class_type": "DGXMonarchInit", "inputs": {"topology": "auto", "mode": "auto"}},
        "2": {"class_type": "DGXMonarchUNETLoader",
              "inputs": {"unet_name": unet, "mesh": ["1", 0]}},
        "3": {"class_type": "EmptySD3LatentImage", "inputs": latent},
        "4": {"class_type": "DGXMonarchKSampler",
              "inputs": {"seed": 1, "steps": 20, "cfg": cfg,
                         "sampler_name": sampler,
                         "model": ["2", 0], "latent_image": ["3", 0]}},
        "5": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "t", "images": ["4", 0]}},
    }


def _rig(tmp_path: Path) -> dict:
    models = tmp_path / "comfy" / "models" / "diffusion_models"
    graphs = tmp_path / "out" / "graphs"
    models.mkdir(parents=True)
    graphs.mkdir(parents=True)
    fp8 = torch.float8_e4m3fn
    for name, unet, video, dtype in (
            ("krea2", "krea2_raw_bf16.safetensors", False, torch.bfloat16),
            ("omnigen2", "omnigen2_bf16.safetensors", False, torch.bfloat16),
            ("pixeldit", "pixeldit_fp8_scaled.safetensors", False, fp8),
            ("h3", "minimax_h3_bf16.safetensors", True, torch.bfloat16)):
        _checkpoint(models, unet, name if name != "h3" else "minimax_h3", dtype)
        (graphs / f"{name}.json").write_text(json.dumps(_graph(unet, video)))
    _checkpoint(models, "krea2_raw_nvfp4.safetensors", "krea2", fp8)
    _checkpoint(models, "orphan_bf16.safetensors", "orphan")
    _checkpoint(models, "stray_bf16.safetensors", "orphan")
    (graphs / "orphan.json").write_text(
        json.dumps(_graph("orphan_bf16.safetensors")))
    (graphs / "cfgone.json").write_text(
        json.dumps(_graph("krea2_raw_bf16.safetensors", cfg=1.0)))
    (graphs / "cfgonepp.json").write_text(
        json.dumps(_graph("krea2_raw_bf16.safetensors", cfg=1.0,
                          sampler="euler_cfg_pp")))
    config = dict(matrix.DEFAULTS)
    config.update(out_dir=tmp_path / "out", comfy_dir=tmp_path / "comfy", model_dir=models,
                  template_dirs=[], exclude=[], toggles={}, toggle_presets={},
                  toggle_combos=[], loras={}, lora_off={}, artifact_groups={},
                  image_presets=PRESETS, video_presets=PRESETS,
                  image_attention=["TORCH_FLASH"], video_attention=["TORCH_FLASH"],
                  sol_families=["minimax_h3", "krea2"])
    return config


def _levers(tmp_path: Path) -> dict:
    """The rig config with one lever of every kind the bar rule sorts.

    Its own config, so no other test's cell set moves under it: a toggle
    multiplies every row of the matrix.
    """
    config = _rig(tmp_path)
    config.update(toggles={"slab_weights": ["on"], "pipeline_depth": [2],
                           "compile_dit": [True]},
                  toggle_combos=[{"comfy_managed": "on", "pipeline_depth": 2},
                                 {"slab_weights": "on", "compile_dit": True}])
    return config


def _pick(cells: list[dict], template: str, preset: str, **want) -> dict:
    hits = [cell for cell in cells if cell["template"] == template
            and cell["preset"] == preset and cell["session"] == "cluster"
            and all(want[key] in str(cell[key]) for key in want)]
    assert hits, f"no {template} {preset} cell for {want}"
    return hits[0]


def _lever(cells: list[dict], template: str, levers: dict,
           preset: str = "auto", **want) -> dict:
    """One cell by its exact lever set: a substring match on the lever dict
    would hand a one-lever ask the combo cell that also carries it."""
    hits = [cell for cell in cells if cell["template"] == template
            and cell["preset"] == preset and cell["session"] == "cluster"
            and cell["levers"] == levers
            and all(want[key] in str(cell[key]) for key in want)]
    assert hits, f"no {template} {preset} cell for {levers} {want}"
    return hits[0]


def test_labels_follow_the_runtime_guards(tmp_path):
    cells, _pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    assert _pick(cells, "omnigen2", "ring2+fsdp", attention="TORCH_FLASH")["label"] == "refuse:K"
    block_scaled = _pick(cells, "krea2", "uly2+fsdp", artifact_quant="nvfp4")
    assert (block_scaled["label"], block_scaled["basis"]) == ("refuse:P", "name-token")
    assert _pick(cells, "krea2", "uly2+fsdp", artifact_quant="bf16")["label"] == "render"
    assert _pick(cells, "pixeldit", "uly2", attention="TORCH_FLASH")["label"] == "refuse:P"
    assert _pick(cells, "h3", "cfg2", attention="TORCH_FLASH")["label"] == "refuse:P"
    assert _pick(cells, "krea2", "uly2", attention="SOL_ATTN_TAU0.7",
                 artifact_quant="bf16")["label"] == "refuse:P"
    waived = _pick(cells, "h3", "uly2", attention="SOL_ATTN_TAU0.7")
    assert (waived["label"], waived["waiver"]) == ("refuse:K", True)


def test_every_example_config_key_reaches_the_loader():
    # A bare key after a [table] header lands in that table, where load_config
    # never looks and the default wins in silence.
    data = tomllib.loads((REPO / "benchmark" / "sweep" / "sweep.example.toml").read_text())
    assert set(data) <= set(matrix.DEFAULTS)
    assert {"sol_kernels", "mem_floor_gib", "video_timeout_s", "toggles"} <= set(data)


def test_the_example_config_reaches_every_committed_template():
    # A template dir the config leaves out is never read, and a LoRA template
    # with no entry runs its lora-off cell only: both cut the sweep down in
    # silence, so the example names every committed folder and every graph that
    # places a loader.
    data = tomllib.loads((REPO / "benchmark" / "sweep" / "sweep.example.toml").read_text())
    templates = REPO / "example_workflows"
    fixtures = REPO / "tests" / "fixtures" / "workflows" / "generated"
    folders = {templates, fixtures}
    paths = sorted(templates.glob("*.json")) + sorted(fixtures.glob("*.json"))
    assert {(REPO / entry).resolve() for entry in data["template_dirs"]} == folders

    excluded = set(data["exclude"])
    wanted = {}
    for path in paths:
        if path.stem in excluded:
            continue
        loaders = sum(1 for node in json.loads(path.read_text())["nodes"]
                      if node["type"] == "DGXMonarchLoraLoader")
        if loaders:
            wanted[path.stem] = loaders
    assert {name: len(names) for name, names in data["loras"].items()} == wanted


# Two nodes' /object_info input blocks, copied from ComfyUI 3216c62e. The type
# strings are what matter: comfy joins a multi-type input's permitted types into
# "FLOAT,INT" and a dynamic combo names an io type of its own, and neither
# matches a plain widget kind.
_OBJECT_INFO = {
    "LTXVEmptyLatentAudio": {"required": {
        "frames_number": ["INT", {"default": 97, "min": 1, "max": 1000}],
        "frame_rate": ["FLOAT,INT", {"default": 25.0, "widgetType": "FLOAT",
                                     "min": 1.0, "max": 1000.0, "step": 0.01}],
        "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
        "audio_vae": ["VAE", {"display_name": "Audio VAE"}]}},
    "SaveAudioAdvanced": {"required": {
        "audio": ["AUDIO", {}],
        "filename_prefix": ["STRING", {"default": "audio/ComfyUI"}],
        "format": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "flac"}]}]}},
}


def test_a_multi_type_widget_keeps_its_name_and_its_slot():
    # A widget the /object_info reader drops is not merely absent: every later
    # widget of that node then takes its neighbour's value. Dropping frame_rate
    # cost the nine LTX audio templates their frame rate and handed batch_size
    # the 24.0 that belonged to it (2026-09-01).
    names = convert.object_info_widget_names(_OBJECT_INFO["LTXVEmptyLatentAudio"])
    assert names == ["frames_number", "frame_rate", "batch_size"]

    ui = {"links": [], "nodes": [{"id": 11, "type": "LTXVEmptyLatentAudio",
                                  "widgets_values": [121, 24.0, 1]}]}
    inputs = convert.convert(ui, lambda node_class: names)["11"]["inputs"]
    assert inputs == {"frames_number": 121, "frame_rate": 24.0, "batch_size": 1}


def test_a_dynamic_combo_is_a_widget_the_reader_places():
    # Same reader, same class of type string: a required widget whose io type
    # is comfy's own name for it, not INT/FLOAT/STRING/BOOLEAN/COMBO.
    names = convert.object_info_widget_names(_OBJECT_INFO["SaveAudioAdvanced"])
    assert names == ["filename_prefix", "format"]


def test_a_linked_create_video_fps_still_consumes_its_widget_slot():
    names = ["fps", "bit_depth", "color_space", "codec"]
    ui = {"links": [[1, 1, 0, 2, 0, "FLOAT"]], "nodes": [
        {"id": 2, "type": "CreateVideo", "inputs": [{"name": "fps", "link": 1}],
         "widgets_values": [30.0, 8, "sRGB", "none"]},
    ]}
    inputs = convert.convert(ui, lambda _node_class: names)["2"]["inputs"]
    assert inputs == {"fps": ["1", 0], "bit_depth": 8,
                      "color_space": "sRGB", "codec": "none"}


def test_the_reader_and_the_generator_name_the_same_widgets():
    # The two maps have to agree, or the sweep sends a different graph
    # depending on whether the driver answered.
    repo_names = convert.repo_widget_names(REPO)
    for node_class, info in _OBJECT_INFO.items():
        assert repo_names(node_class) == convert.object_info_widget_names(info)


def test_cell_ids_are_stable_and_unmatched_files_are_reported(tmp_path):
    config = _rig(tmp_path)
    first, _pruned, unmatched = matrix.build_cells(config)
    second, _again, _rest = matrix.build_cells(config)
    assert [cell["id"] for cell in first] == [cell["id"] for cell in second]
    assert len({cell["id"] for cell in first}) == len(first)
    assert "stray_bf16.safetensors" in unmatched


def test_reference_specs_name_a_bar_and_a_source(tmp_path):
    cells, _pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    assert _pick(cells, "krea2", "uly2+fsdp", artifact_quant="bf16")["reference"] == {
        "kind": "resident-leg", "preset": "uly2", "bar": "bit-identical"}
    topology = _pick(cells, "krea2", "uly2", artifact_quant="bf16", attention="TORCH_FLASH")
    local = [cell for cell in cells if cell["id"] == topology["reference"]["cell"]]
    assert local and local[0]["session"] == "local" and local[0]["mode"] == "local"
    assert topology["reference"]["probe_steps"] == 1


def test_every_probe_reference_renders_the_probe_leg_it_is_compared_against(tmp_path):
    # The candidate compares its probe leg to sweep_<reference>_probe, so the
    # reference has to plan that leg, not just carry probe_steps. The lever rig
    # is used because a numeric lever cell plans a probe leg of its own, and the
    # auto cell of its row has to answer it.
    cells, _pruned, _unmatched = matrix.build_cells(_levers(tmp_path))
    by_id = {cell["id"]: cell for cell in cells}
    asked = levered = 0
    for cell in cells:
        reference = cell["reference"]
        if reference["kind"] != "cell" or not (reference.get("probe_steps") and cell["probe"]):
            continue
        asked += 1
        levered += bool(cell["levers"])
        legs = [leg for leg, *_rest in run.leg_plan(by_id[reference["cell"]])]
        assert "probe" in legs, cell["id"]
    assert asked and levered


def _payload(message: str, kind: str = "builtins.RuntimeError") -> dict:
    return {"status": {"status_str": "error", "messages": [
        ["execution_error", {"exception_message": message, "exception_type": kind}]]}}


def test_outcome_classifier_reads_the_message_not_the_type():
    unsupported = "dgx_monarch.adapters.base.UnsupportedModelError"
    tagged = _payload("[dgxm:P] cfg-parallel has nothing to split", unsupported)
    assert driver.classify_history(tagged) == ("refuse:P", driver.error_text(tagged))
    # An opening the runtime still raises without a tag (the FSDP launch
    # precision refusals took the class P card on 2026-09-09).
    untagged = _payload("FSDP: no shardable block lists found on the model.", unsupported)
    assert driver.classify_history(untagged)[0] == "refuse:untyped"
    crash = _payload("CUDA error: an illegal memory access was encountered",
                     "dgx_monarch.actor.failure.CudaContextPoisonedError")
    assert driver.classify_history(crash)[0] == "crash"
    # The type is secondary evidence, never the discriminator: a crash whose
    # type names a refusal class stays a crash.
    assert driver.classify_history(
        _payload("CUDA error: device-side assert triggered", unsupported))[0] == "crash"
    assert driver.classify_history({"status": {"completed": True}}) == ("render", "")
    assert driver.classify_history(None, timed_out=True)[0] == "timeout"


def test_every_untagged_refusal_the_runtime_raises_is_recognised():
    # The classifier walks the runtime's raise sites instead of carrying a
    # copy of their texts, so a guard added upstream cannot land as a crash.
    # The texts below include walked openings and the UNTAGGED_LITERALS ones a
    # bare ValueError, RuntimeError or formatted table carries.
    tree = ast.parse((REPO / "src" / "dgx_monarch" / "nodes" / "auto_gate.py").read_text())
    reasons = next(node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], "id", "") == "_FSDP_DENIAL_REASONS")
    denials = [value.value.split("{")[0] for value in reasons.values]
    assert len(denials) >= 5
    texts = [*denials,
             "automatic FSDP Gate context/ledger lookup failed",
             "automatic FSDP execution is unavailable for a dual-model request "
             "because the clean-reload Gate proves one model only",
             "explicit topology preset 'dp2' on world 2 preserves its literal "
             "model-parallel degree and therefore leaves dp2, but latent batch 1 "
             "is not divisible by 2.",
             "activation footprint preflight refuses ltx.safetensors: estimated 90 GiB",
             "stock residency cannot load wan.safetensors: weights are 40 GiB",
             "omnigen2 has 21 attention heads (7 kv), not divisible by the "
             "requested ulysses degree 2",
             "cfg-parallel got a model call of batch=1, which does not split into "
             "2 equal slices.",
             "comfy_managed=on and slab_weights=on are two different residencies "
             "for the same weights.",
             "comfy_managed=on forces lora_low_rss off (docs/TROUBLESHOOTING.md #62);"]
    for text in texts:
        assert driver.classify_history(_payload(text))[0] == "refuse:untyped", text


def test_the_fsdp_proof_abort_is_read_as_a_refusal_not_a_crash():
    """Both FSDP proof stops raise where the AST walk cannot read them.

    The incomplete-proof stop raises through ``runtime["FsdpGateProofError"]``
    and the abort through a rebound name, so UNTAGGED_LITERALS carries the
    opening clause both share. The abort appends what stopped the proof after
    that opening, so the opening still matches.
    """
    assert "FSDP clean-reload proof " in driver.untagged_prefixes()
    aborted = ("FsdpGateProofError: FSDP clean-reload proof aborted before a "
               "terminal verdict: RuntimeError: boom")
    incomplete = ("FsdpGateProofError: FSDP clean-reload proof is incomplete: "
                  "FSDP reload cycle did not report every rank")
    for text in (aborted, incomplete):
        assert driver.classify_history(_payload(text))[0] == "refuse:untyped", text


def test_a_worker_refusal_survives_its_actor_error_framing():
    # A worker's refusal reaches /history inside monarch's ActorError, whose
    # text prints the remote type name ahead of the clause. Only the tag search
    # survives that; the opening clause has to be read past the framing.
    clause = ("cfg-parallel got a model call of batch=1, which does not split "
              "into 2 equal slices.")
    for name in ("RuntimeError", "dgx_monarch.adapters.base.UnsupportedModelError"):
        framed = f"A remote actor call has failed.\n {name}: {clause}\n"
        assert driver.classify_history(_payload(framed))[0] == "refuse:untyped", framed
    # That clause carries a class P tag since 2026-09-01, and the tag search
    # reads it whatever the framing. The untagged form above stays pinned: it
    # is what every sweep log written before the tag carries, and driver.py
    # keeps the literal that reads those logs.
    from dgx_monarch.refusal import RefusalClass, refusal

    tagged = refusal(RefusalClass.PHYSICS, clause)
    for name in ("RuntimeError", "dgx_monarch.adapters.base.UnsupportedModelError"):
        framed = f"A remote actor call has failed.\n {name}: {tagged}\n"
        assert driver.classify_history(_payload(framed))[0] == "refuse:P", framed
    crash = ("A remote actor call has failed.\n RuntimeError: CUDA error: an "
             "illegal memory access was encountered\n")
    assert driver.classify_history(_payload(crash))[0] == "crash"


def test_frame_sets_compare_by_pixels_and_count(tmp_path):
    from PIL import Image

    def frame(name: str, value: int) -> Path:
        path = tmp_path / name
        Image.new("RGB", (4, 4), (value, value, value)).save(path)
        return path

    same = compare.compare_frames([frame("a_1.png", 40)], [frame("b_1.png", 40)])
    assert same["identical"] is True and same["nrms"] == 0.0 and same["frames"] == 1
    moved = compare.compare_frames([frame("c_1.png", 44)], [frame("b_1.png", 40)])
    assert moved["max_abs"] == 4 and moved["nrms"] > 0
    short = compare.compare_frames([frame("d_1.png", 40)],
                                   [frame("b_1.png", 40), frame("e_2.png", 40)])
    assert "frame count" in short["error"]


def test_a_cfg_topology_on_a_cfg_one_graph_is_a_refusal(tmp_path):
    # ComfyUI runs no uncond pass at cfg 1.0, so the batched wrapper a cfg
    # topology installs never gets the two slices it splits.
    cells, _pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    assert _pick(cells, "cfgone", "cfg2")["label"] == "refuse:P"
    # The same graph under fsdp keeps the untagged label: the FSDP gate denies
    # the run before the cfg-one check can tag it.
    assert _pick(cells, "cfgone", "cfg2+fsdp")["label"] == "refuse:untyped"
    assert _pick(cells, "krea2", "cfg2", artifact_quant="bf16")["label"] == "render"


def test_a_cfg_pp_sampler_at_cfg_one_still_renders_on_a_cfg_topology(tmp_path):
    # Measured 2026-09-02: krea2 at cfg2 with CFG 1.0 rendered three legs
    # while flux1 at cfg2 with CFG 1.0 refused class P. The two graphs differ
    # by sampler, not by family. A cfg++ sampler sets
    # disable_cfg1_optimization, so comfy keeps the unconditional pass and
    # hands the wrapper a batch of 2 that both ranks split. Reading CFG alone
    # mislabels every cfg++ cell.
    cells, _pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    assert _pick(cells, "cfgonepp", "cfg2", artifact_quant="bf16")["label"] == "render"
    assert _pick(cells, "cfgone", "cfg2", artifact_quant="bf16")["label"] == "refuse:P"
    # The sampler is the deciding fact, so the cell carries it as evidence.
    assert matrix.sampler_keeps_uncond("euler_cfg_pp")
    assert not matrix.sampler_keeps_uncond("euler")


def test_a_dp_cell_whose_batch_does_not_split_takes_the_class_the_driver_raises(tmp_path):
    # The loader rejects a batch-1 dp2 graph before the footprint card or any
    # checkpoint load. The matrix must call the same check to label it class P.
    from dgx_monarch.nodes.render_validation import LITERAL_BATCH_LATENT_CLASSES

    config = _rig(tmp_path)
    config.update(image_presets=["auto", "dp2"], video_presets=["auto"],
                  toggles={"batch_size": [2]}, toggle_presets={"batch_size": ["dp2"]})
    # A latent source the loader node cannot read states no literal, so the
    # same refusal arrives at the render submit instead, with the real tensor.
    unpriced = _graph("krea2_raw_bf16.safetensors")
    unpriced["3"]["class_type"] = "CustomLatentSource"
    (config["out_dir"] / "graphs" / "unpriced.json").write_text(json.dumps(unpriced))
    cells, _pruned, _unmatched = matrix.build_cells(config)
    hoisted = _pick(cells, "krea2", "dp2", artifact_quant="bf16", batch="1")
    assert (hoisted["label"], hoisted["basis"]) == ("refuse:P", "guard")
    assert _pick(cells, "unpriced", "dp2", batch="1")["label"] == "refuse:P"
    assert "EmptySD3LatentImage" in LITERAL_BATCH_LATENT_CLASSES
    assert "CustomLatentSource" not in LITERAL_BATCH_LATENT_CLASSES
    # The batch a toggle moved is the batch the runner queues, so the graph the
    # check reads carries it too and a batch of 2 splits across dp2.
    split = _pick(cells, "krea2", "dp2", artifact_quant="bf16", batch="2")
    assert split["batch_toggled"] and split["label"] == "render"


def test_the_adapter_refuses_at_the_load_before_the_cfg_one_shortcut(tmp_path):
    # A family with no cfg-parallel support refuses when the worker binds its
    # adapter; the cfg 1.0 wrapper only trips later, at the sampler.
    config = _rig(tmp_path)
    (config["out_dir"] / "graphs" / "pidone.json").write_text(
        json.dumps(_graph("pixeldit_fp8_scaled.safetensors", cfg=1.0)))
    cells, _pruned, _unmatched = matrix.build_cells(config)
    assert _pick(cells, "pidone", "cfg2")["label"] == "refuse:P"
    assert _pick(cells, "cfgone", "cfg2")["label"] == "refuse:P"


def test_comfy_managed_refuses_before_the_live_quant_is_ever_read(tmp_path):
    # Comfy-managed residency under FSDP is refused before any weight is read
    # (the loader node's preflight, then actor/store_fsdp.py on the worker), so
    # the block-scaled refusal later in the load cannot be the one that fires.
    config = _rig(tmp_path)
    config.update(toggles={"comfy_managed": ["on"]},
                  toggle_presets={"comfy_managed": ["uly2+fsdp"]})
    cells, _pruned, _unmatched = matrix.build_cells(config)
    managed = [cell for cell in cells if cell["levers"].get("comfy_managed") == "on"
               and cell["template"] == "krea2" and "nvfp4" in cell["artifact_quant"]]
    assert managed and {cell["label"] for cell in managed} == {"refuse:P"}


def test_an_auto_row_that_cannot_resolve_labels_rather_than_aborting(tmp_path):
    # choose_auto_topology refuses like any other guard. Raising through
    # build_cells would end the run with no cells.jsonl at all.
    config = _rig(tmp_path)
    config["world"] = 4
    # Keep the tiny graph and switch only its header signature to Chroma bf16,
    # whose auto row at 1 MP is cfg2. At world 4 that row cannot fold its spare
    # ranks, so auto refuses, and the matrix must label that refusal.
    _checkpoint(config["model_dir"], "krea2_raw_bf16.safetensors", "chroma")
    cells, _pruned, _unmatched = matrix.build_cells(config)
    auto = [cell for cell in cells if cell["preset"] == "auto" and cell["template"] == "krea2"
            and cell["session"] == "cluster"]
    assert auto and {cell["label"] for cell in auto} == {"refuse:K"}


def test_a_lora_name_count_mismatch_is_printed_and_the_arm_dropped(tmp_path):
    # A short list would delete loaders from the graph and score the row as a
    # stack the template never places.
    config = _rig(tmp_path)
    graph = _graph("krea2_raw_bf16.safetensors")
    for node_id in ("6", "7"):
        graph[node_id] = {"class_type": "DGXMonarchLoraLoader",
                          "inputs": {"model": ["2", 0], "lora_name": "placeholder.safetensors",
                                     "strength_model": 0.8}}
    (config["out_dir"] / "graphs" / "twolora.json").write_text(json.dumps(graph))
    config["loras"] = {"twolora": ["one.safetensors"]}
    cells, pruned, _unmatched = matrix.build_cells(config)
    assert any("twolora: 2 LoRA loader(s) against 1" in line for line in pruned)
    assert not [cell for cell in cells if cell["loras"]]


def test_text_encoder_loader_classes_pin_against_the_artifact_manifest_widgets():
    # tools/check_artifacts.py LOADER_WIDGETS is the manifest's own record of
    # which widget on which node class names a text-encoder file, and the
    # models subfolder it resolves against; matrix.py must read the same pair
    # for every CLIP loader class it prices.
    spec = importlib.util.spec_from_file_location(
        "dgxm_check_artifacts_for_te_pin", REPO / "tools" / "check_artifacts.py")
    check_artifacts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_artifacts)
    for node_class, widgets in matrix.TEXT_ENCODER_LOADER_CLASSES.items():
        assert check_artifacts.LOADER_WIDGETS[node_class] == \
            dict.fromkeys(widgets, "text_encoders")


def test_resident_text_encoder_gib_reads_every_clip_loader_widget(tmp_path):
    te_dir = tmp_path / "comfy" / "models" / "text_encoders"
    te_dir.mkdir(parents=True)
    (te_dir / "clip_l.safetensors").write_bytes(b"x" * 2**20)
    (te_dir / "t5xxl.safetensors").write_bytes(b"x" * 2**20)
    graph = {
        "1": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl.safetensors"}},
        "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_l.safetensors"}},
    }
    # The DualCLIPLoader's two files count; the CLIPVisionLoader is not priced.
    assert matrix.resident_text_encoder_gib(graph, tmp_path / "comfy") == \
        round(2 * 2**20 / 2**30, 1)
    # A name the folder does not hold prices as absent, never guessed.
    missing = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "nowhere.safetensors"}}}
    assert matrix.resident_text_encoder_gib(missing, tmp_path / "comfy") == 0.0


def test_a_local_cell_prices_the_resident_text_encoder():
    # Kandinsky5-Video Pro's 40.4 GiB bf16 DiT alone reads under the 50 GiB
    # flag, but its resident text encoder tipped a local cell's real load over
    # it: the local cell refused class C at stock_load_preflight with no
    # capacity_risk flag to score it PASS-capacity (matrix fixed 2026-09-28).
    assert not matrix.capacity_risk_for("krea2", 40.4)
    assert not matrix.capacity_risk_for("krea2", 40.4, session="local")
    assert matrix.capacity_risk_for("krea2", 40.4, session="local", resident_te_gib=10.0)
    assert matrix.capacity_risk_basis_for("krea2", 40.4, session="local", resident_te_gib=10.0) \
        == "local-stock-load-priced-with-resident-text-encoder"
    # A cluster cell puts the loader on the driver host, not the worker this
    # price charges, so the same bytes buy no flag there.
    assert not matrix.capacity_risk_for("krea2", 40.4, session="cluster", resident_te_gib=10.0)
    assert matrix.capacity_risk_basis_for(
        "krea2", 40.4, session="cluster", resident_te_gib=10.0) is None
    # A DiT already over the line needs no help from the text encoder, so the
    # basis marker covers only the case that newly crosses it: on every local
    # cell of 50 GiB or more it would re-run settled records for nothing
    # (capacity_risk_basis_for docstring).
    assert matrix.capacity_risk_for("krea2", 60.0, session="local", resident_te_gib=10.0)
    assert matrix.capacity_risk_basis_for(
        "krea2", 60.0, session="local", resident_te_gib=10.0) is None


def test_a_local_kandinsky_shaped_cell_flags_capacity_risk_end_to_end(tmp_path):
    # The same boundary, through build_cells: a real (tiny) checkpoint and a
    # real (tiny) CLIP file, both truncated up to a size that only crosses the
    # 50 GiB line once summed, on a local cell only.
    config = _rig(tmp_path)
    te_dir = config["comfy_dir"] / "models" / "text_encoders"
    te_dir.mkdir(parents=True)
    (te_dir / "qwen3vl.safetensors").write_bytes(b"x")
    os.truncate(te_dir / "qwen3vl.safetensors", 10 * 2**30)
    graph = _graph("krea2_raw_bf16.safetensors")
    graph["6"] = {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": "qwen3vl.safetensors", "type": "krea2"}}
    (config["out_dir"] / "graphs" / "tewired.json").write_text(json.dumps(graph))
    os.truncate(config["model_dir"] / "krea2_raw_bf16.safetensors", 44 * 2**30)
    cells, _pruned, _unmatched = matrix.build_cells(config)
    local = next(cell for cell in cells if cell["template"] == "tewired"
                and cell["preset"] == "single" and cell["session"] == "local"
                and cell["artifact_quant"] == "bf16")
    assert local["capacity_risk"] and local["capacity_risk_basis"] == \
        "local-stock-load-priced-with-resident-text-encoder"
    cluster = [cell for cell in cells if cell["template"] == "tewired"
              and cell["preset"] == "cfg2" and cell["session"] == "cluster"
              and cell["artifact_quant"] == "bf16"]
    assert cluster and not cluster[0]["capacity_risk"]


def test_a_lora_cell_on_an_unadmitted_fsdp_checkpoint_labels_class_p(tmp_path):
    # adapters/fsdp_lora_admission.py refuses, typed and from the header alone,
    # a LoRA stack on a comfy-kitchen quantized checkpoint under FSDP. A matrix
    # that does not call it labels such a cell render, and the rig then refuses
    # it class P (matrix fixed 2026-09-28).
    config = _rig(tmp_path)
    _checkpoint(config["model_dir"], "krea2_raw_int8.safetensors", "krea2", torch.int8)
    graph = _graph("krea2_raw_bf16.safetensors")
    graph["6"] = {"class_type": "DGXMonarchLoraLoader",
                  "inputs": {"model": ["2", 0], "lora_name": "placeholder.safetensors",
                             "strength_model": 0.8}}
    (config["out_dir"] / "graphs" / "krea2lora.json").write_text(json.dumps(graph))
    config["loras"] = {"krea2lora": ["one.safetensors"]}
    cells, _pruned, _unmatched = matrix.build_cells(config)
    at_preset = [cell for cell in cells if cell["template"] == "krea2lora"
                and cell["preset"] == "uly2+fsdp" and cell["session"] == "cluster"]
    int8_with_lora = next(c for c in at_preset if c["loras"] and c["artifact_quant"] == "int8")
    int8_without_lora = next(c for c in at_preset
                             if not c["loras"] and c["artifact_quant"] == "int8")
    bf16_with_lora = next(c for c in at_preset if c["loras"] and c["artifact_quant"] == "bf16")
    assert int8_with_lora["fsdp_lora_admission"] == "quantized_shards"
    assert (int8_with_lora["label"], int8_with_lora["basis"]) == ("refuse:P", "guard")
    # Without a LoRA stack the same checkpoint keeps its own quant-only answer:
    # int8 is admitted for a plain FSDP launch, so nothing here refuses.
    assert int8_without_lora["fsdp_lora_admission"] == "quantized_shards"
    assert int8_without_lora["label"] == "render"
    # bf16 carries none of the comfy-kitchen markers, so the property reads
    # None and the cell keeps whatever the lora_low_rss lever already decided.
    assert bf16_with_lora["fsdp_lora_admission"] is None
    assert bf16_with_lora["label"] == "render"


def test_a_lora_patching_an_fp32_stored_weight_labels_class_p_under_fsdp(tmp_path):
    # actor/fsdp_lora.py's in-bake check refuses class P when a patched weight is
    # stored fp32 but runs bf16: krea2_raw_bf16 with its turbo LoRA did that on
    # uly2+fsdp in the 2026-09-29 debt re-run while the matrix said render.
    config = _rig(tmp_path)
    # A bf16 core with a small fp32 island, well under the launch contract's 5% share.
    save_file({"txtfusion.a.weight": torch.zeros(64, 64, dtype=torch.bfloat16),
               "tproj.1.weight": torch.zeros(2, 2, dtype=torch.float32)},
              str(config["model_dir"] / "krea2_raw_bf16.safetensors"))
    loras = config["comfy_dir"] / "models" / "loras"
    loras.mkdir(parents=True, exist_ok=True)
    save_file({"diffusion_model.tproj.1.lora_down.weight": torch.zeros(1, 2),
               "diffusion_model.tproj.1.lora_up.weight": torch.zeros(2, 1)},
              str(loras / "fp32.safetensors"))
    save_file({"diffusion_model.txtfusion.a.lora_down.weight": torch.zeros(1, 2),
               "diffusion_model.txtfusion.a.lora_up.weight": torch.zeros(2, 1)},
              str(loras / "bf16.safetensors"))
    graph = _graph("krea2_raw_bf16.safetensors")
    graph["6"] = {"class_type": "DGXMonarchLoraLoader",
                  "inputs": {"model": ["2", 0], "lora_name": "placeholder.safetensors",
                             "strength_model": 0.8}}
    for name in ("fp32lora", "bf16lora"):
        (config["out_dir"] / "graphs" / f"{name}.json").write_text(json.dumps(graph))
    config["loras"] = {"fp32lora": ["fp32.safetensors"], "bf16lora": ["bf16.safetensors"]}
    config["toggles"] = {"lora_low_rss": ["off"]}
    config["toggle_presets"] = {"lora_low_rss": ["uly2+fsdp"]}
    cells, _pruned, _unmatched = matrix.build_cells(config)

    def pick(template, preset):
        return next(c for c in cells if c["template"] == template and c["preset"] == preset
                    and c["session"] == "cluster" and c["loras"] and c["artifact_quant"] == "bf16"
                    and not c["levers"])

    patched = pick("fp32lora", "uly2+fsdp")
    assert patched["fsdp_lora_f32_patched"] == 1
    assert (patched["label"], patched["basis"]) == ("refuse:P", "header")
    # A resident topology never runs the FSDP bake, and a bf16-only stack passes it.
    assert pick("fp32lora", "uly2")["label"] == "render"
    assert pick("bf16lora", "uly2+fsdp")["fsdp_lora_f32_patched"] == 0
    assert pick("bf16lora", "uly2+fsdp")["label"] == "render"
    # With lora_low_rss off the launch guard answers first, so the basis is that guard.
    low_rss_off = next(c for c in cells if c["template"] == "fp32lora" and c["preset"] == "uly2+fsdp"
                       and c["loras"] and c["levers"].get("lora_low_rss") == "off")
    assert (low_rss_off["label"], low_rss_off["basis"]) == ("refuse:P", "guard")


def test_an_fp32_patch_on_a_family_not_measured_to_cast_still_renders(tmp_path):
    # Whether comfy casts a stored fp32 weight is family model code; only a
    # family measured to cast (krea2) takes the header label.
    config = _rig(tmp_path)
    save_file({"double_stream_layers.0.weight": torch.zeros(64, 64, dtype=torch.bfloat16),
               "img_instruct_attn.a.weight": torch.zeros(64, 64, dtype=torch.bfloat16),
               "tproj.1.weight": torch.zeros(2, 2, dtype=torch.float32)},
              str(config["model_dir"] / "boogu_bf16.safetensors"))
    loras = config["comfy_dir"] / "models" / "loras"
    loras.mkdir(parents=True, exist_ok=True)
    save_file({"diffusion_model.tproj.1.lora_down.weight": torch.zeros(1, 2),
               "diffusion_model.tproj.1.lora_up.weight": torch.zeros(2, 1)},
              str(loras / "fp32.safetensors"))
    graph = _graph("boogu_bf16.safetensors")
    graph["6"] = {"class_type": "DGXMonarchLoraLoader",
                  "inputs": {"model": ["2", 0], "lora_name": "placeholder.safetensors",
                             "strength_model": 0.8}}
    (config["out_dir"] / "graphs" / "booglora.json").write_text(json.dumps(graph))
    config["loras"] = {"booglora": ["fp32.safetensors"]}
    cells, _pruned, _unmatched = matrix.build_cells(config)
    cell = next(c for c in cells if c["template"] == "booglora" and c["preset"] == "uly2+fsdp"
                and c["session"] == "cluster" and c["loras"] and not c["levers"])
    assert cell["fsdp_lora_f32_patched"] == 1
    assert cell["basis"] != "header"


def test_a_template_naming_media_the_rig_lacks_is_pruned_not_swept(tmp_path):
    """Comfy rejects such a graph at validation, so its cells can only reject.

    Three shipped templates name user media. They derived a full block of cells
    each, and every one came back a graph rejection rather than an answer
    (pruned since 2026-09-09).
    """
    config = _rig(tmp_path)
    inputs = config["comfy_dir"] / "input"
    inputs.mkdir()
    (inputs / "dgxm_test_portrait.png").write_bytes(b"x")
    graph = _graph("krea2_raw_bf16.safetensors")
    graph["90"] = {"class_type": "LoadVideo", "inputs": {"file": "source_video.mp4"}}
    graph["91"] = {"class_type": "LoadImage", "inputs": {"image": "blue_dancer.png"}}
    (config["out_dir"] / "graphs" / "usermedia.json").write_text(json.dumps(graph))
    twin = _graph("krea2_raw_bf16.safetensors")
    twin["90"] = {"class_type": "LoadImage",
                  "inputs": {"image": "dgxm_test_portrait.png [input]"}}
    (config["out_dir"] / "graphs" / "testmedia.json").write_text(json.dumps(twin))

    cells, pruned, _unmatched = matrix.build_cells(config)

    note = [line for line in pruned if line.startswith("usermedia:")]
    assert note and "blue_dancer.png, source_video.mp4" in note[0]
    assert not [cell for cell in cells if cell["template"] == "usermedia"]
    # The twin on synthetic media is the row that runs, annotation and all.
    assert [cell for cell in cells if cell["template"] == "testmedia"]


def test_a_media_folder_that_is_not_there_prunes_nothing(tmp_path):
    """Every name would read as missing, and the sweep would prune itself."""
    config = _rig(tmp_path)
    graph = _graph("krea2_raw_bf16.safetensors")
    graph["90"] = {"class_type": "LoadVideo", "inputs": {"file": "source_video.mp4"}}
    # A widget converted to an input carries a link, not a name.
    graph["91"] = {"class_type": "LoadImage", "inputs": {"image": ["90", 0]}}
    (config["out_dir"] / "graphs" / "usermedia.json").write_text(json.dumps(graph))

    cells, pruned, _unmatched = matrix.build_cells(config)

    assert not [line for line in pruned if "input folder does not hold" in line]
    assert [cell for cell in cells if cell["template"] == "usermedia"]
    assert matrix.missing_media(graph, config["comfy_dir"] / "input") == []


def test_a_remote_command_travels_as_one_quoted_string(monkeypatch):
    # ssh joins its argv with spaces and the far shell re-parses it, so an
    # unquoted pipe runs there and the reading comes back empty.
    seen: dict = {}

    def fake_run(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, "MemAvailable: 2097152 kB\n", "")

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    values, error = run._mem_gib("sibling")
    assert not error and values["MemAvailable"] == 2.0
    assert seen["command"][:2] == ["ssh", "-n"]
    # One grep per sample, and the whole pattern crosses as one quoted word:
    # the alternation, the anchor and the escaped parens the LRU field names
    # need all read as shell metacharacters otherwise.
    assert seen["command"][-1] == f"grep -E '{run.MEMINFO_PATTERN}' /proc/meminfo"
    assert r"Active\(anon\)" in seen["command"][-1]
    assert seen["kwargs"]["stdin"] is subprocess.DEVNULL


def test_a_box_that_gives_no_reading_is_recorded_not_waited_on(monkeypatch):
    monkeypatch.setattr(run, "_on", lambda host, command, timeout: ("", "rc 127: not found"))
    floor = run.wait_for_memory(["", "sibling"], 100.0, 600.0)
    assert floor["cleared"] is False and set(floor["errors"]) == {"head", "sibling"}


def test_a_retained_attempt_is_not_a_summary_row(tmp_path):
    from benchmark.sweep import report

    cells = tmp_path / "cells"
    cells.mkdir()
    record = {"cell": {"session": "cluster", "id": "a"}, "pin": {"torch": "2.9"}}
    (cells / "a.json").write_text(json.dumps(record))
    stale = dict(record, pin={"torch": "2.8"})
    (cells / "a.attempt1.json").write_text(json.dumps(stale))
    rows = report.records_for(tmp_path, "cluster")
    assert [entry["pin"]["torch"] for entry in rows] == ["2.9"]


def _ideogram4(config, tmp_path, dual: bool) -> dict:
    name = "ideogram4_bf16.safetensors"
    _checkpoint(config["model_dir"], name, "ideogram4")
    graph = _graph(name)
    if dual:
        graph["6"] = {"class_type": "DGXMonarchUncondUNETLoader",
                      "inputs": {"unet_name": name, "mesh": ["1", 0]}}
    (config["out_dir"] / "graphs" / f"ig4{'dual' if dual else 'solo'}.json").write_text(
        json.dumps(graph))
    return config


def test_dual_model_cfg2_needs_the_second_checkpoint_the_graph_places(tmp_path):
    # The worker grants dual-model cfg2 from the adapter flag alone. The sample
    # path then loads one checkpoint per rank, so a graph that places only one
    # checkpoint refuses class P.
    config = _rig(tmp_path)
    _ideogram4(config, tmp_path, dual=False)
    _ideogram4(config, tmp_path, dual=True)
    cells, _pruned, _unmatched = matrix.build_cells(config)
    assert _pick(cells, "ig4solo", "cfg2")["label"] == "refuse:P"
    assert _pick(cells, "ig4dual", "cfg2")["label"] == "render"
    # Never under FSDP, on either graph.
    assert _pick(cells, "ig4dual", "uly2+fsdp")["label"] != "render"
    # A cfg that splits by checkpoint never asks whether cond and uncond batch,
    # so no prompt count is owed and the basis says the guards decided.
    config.update(toggles={"auto_gate": ["off"]},
                  toggle_presets={"auto_gate": ["cfg2+fsdp"]})
    cells, _pruned, _unmatched = matrix.build_cells(config)
    gated = _pick(cells, "ig4dual", "cfg2+fsdp", auto_gate="off")
    assert gated["basis"] == "guard"


def test_comfy_managed_dual_model_cells_refuse_before_loading(tmp_path):
    config = _rig(tmp_path)
    _ideogram4(config, tmp_path, dual=False)
    _ideogram4(config, tmp_path, dual=True)
    config.update(
        toggles={"comfy_managed": ["on"]},
        toggle_presets={"comfy_managed": ["auto", "uly2", "cfg2", "ring2"]},
    )

    cells, _pruned, _unmatched = matrix.build_cells(config)

    for preset in ("auto", "uly2", "cfg2", "ring2"):
        assert _lever(cells, "ig4dual", {"comfy_managed": "on"}, preset)["label"] == "refuse:P"
    assert _lever(cells, "ig4dual", {}, "auto")["label"] == "render"
    assert _lever(cells, "ig4solo", {"comfy_managed": "on"}, "auto")["label"] == "render"


def test_kernels_that_resolve_to_one_collapse_to_one_cell(tmp_path):
    # An auto row with sage on rewrites every declared kernel, so the three
    # attention cells at preset auto would queue the same render three times.
    config = _rig(tmp_path)
    _checkpoint(config["model_dir"], "krea2_raw_fp8_scaled.safetensors", "krea2",
                torch.float8_e4m3fn)
    (config["out_dir"] / "graphs" / "big.json").write_text(json.dumps(
        _graph("krea2_raw_fp8_scaled.safetensors") | {"3": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1536, "height": 1536, "batch_size": 1}}}))
    config["image_attention"] = ["TORCH_FLASH", "TORCH_CUDNN", "SAGE_AUTO"]
    config["sol_families"] = []
    cells, pruned, _unmatched = matrix.build_cells(config)
    auto = [cell for cell in cells if cell["template"] == "big" and cell["preset"] == "auto"
            and cell["session"] == "cluster" and not cell["levers"]
            and "fp8" in cell["artifact_quant"]]
    assert [cell["resolved_attention"] for cell in auto] == ["SAGE_AUTO"]
    # The bf16 sibling keeps all three: its row asks for no rewrite.
    bf16 = [cell for cell in cells if cell["template"] == "big" and cell["preset"] == "auto"
            and cell["session"] == "cluster" and not cell["levers"]
            and cell["artifact_quant"] == "bf16"]
    assert len(bf16) == 3
    assert any("collapsed onto a twin" in line for line in pruned)
    # A reference never points at a cell the collapse removed.
    ids = {cell["id"] for cell in cells}
    assert all(cell["reference"].get("cell", next(iter(ids))) in ids for cell in cells)


def test_an_artifact_groups_array_names_the_base_loaders_extras(tmp_path):
    config = _rig(tmp_path)
    config["artifact_groups"] = {"krea2": ["krea2_raw_nvfp4.safetensors"]}
    cells, _pruned, _unmatched = matrix.build_cells(config)
    assert {cell["artifact_quant"] for cell in cells if cell["template"] == "krea2"} == \
        {"bf16", "nvfp4"}
    config["artifact_groups"] = {"krea2": "one name"}
    with pytest.raises(matrix.ConfigError, match="krea2"):
        matrix.build_cells(config)


def test_a_family_that_cannot_split_its_heads_refuses_ulysses(tmp_path):
    cells, _pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    assert _pick(cells, "omnigen2", "uly2")["label"] == "refuse:untyped"
    assert _pick(cells, "omnigen2", "ring2")["label"] == "render"


class _Args:
    def __init__(self, session: str, cells: str = "all", limit: int = 0,
                 include_waivers: bool = False) -> None:
        self.session, self.cells, self.limit = session, cells, limit
        self.include_waivers = include_waivers


def test_an_unreadable_family_is_skipped_rather_than_swept(tmp_path):
    # Family gates are all disarmed on an unknown family, so a render there
    # proves nothing. The row says so, prints why, and never reaches a run.
    cells, pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    orphan = [cell for cell in cells if cell["template"] == "orphan"]
    assert orphan and {cell["label"] for cell in orphan} == {"skip:unknown-family"}
    assert any("orphan: the header sniff names no family" in line for line in pruned)
    chosen, _cut = run.select(cells, _Args(session="cluster"))
    assert not [cell for cell in chosen if cell["template"] == "orphan"]


def test_selection_records_every_cell_it_cuts(tmp_path):
    cells, _pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    chosen, pruned = run.select(cells, _Args(session="cluster"))
    assert {"sampled_out", "video_beyond_first", "skipped"} <= set(pruned)
    cut = sum(len(ids) for ids in pruned.values())
    cluster = [cell for cell in cells if cell["session"] == "cluster"]
    assert len(chosen) + cut == len(cluster)
    assert not [cell for cell in chosen if cell["label"].startswith("skip:")]


def test_journal_lines_preserve_attention_and_unavailable_evidence(monkeypatch):
    monkeypatch.setattr(
        run,
        "_on",
        lambda *_args: (
            "ordinary worker chatter\n"
            "USP attention kernel: TORCH_FLASH\n"
            "attention backend unavailable (not installed)\n"
            "unavailable (the selected kernel cannot run)\n"
            "identity gate PASS\n",
            "rc 255: transport lost",
        ),
    )

    assert run.journal_lines("spark-2", "2026-08-30 00:00:00") == [
        "USP attention kernel: TORCH_FLASH",
        "attention backend unavailable (not installed)",
        "unavailable (the selected kernel cannot run)",
        "identity gate PASS",
        "journal read failed: rc 255: transport lost",
    ]


def test_quant_siblings_group_across_separators_and_residue_words():
    same = {"ltx-2.5-22b-dev-transformer-bf16",
            "ltx-2.5-22b-dev-transformer-comfy-int8-convrot"}
    assert len({matrix.split_quant(stem)[0] for stem in same}) == 1
    assert matrix.split_quant("Chroma1-HD-fp8mixed-final")[0] == \
        matrix.split_quant("Chroma1-HD")[0]
    assert matrix.split_quant("krea2_turbo_bf16")[0] != matrix.split_quant("lens_bf16")[0]


def test_an_extra_artifact_joins_the_loader_the_config_names():
    from pathlib import Path as _Path

    unets = {"2": "ideogram4_fp8_scaled.safetensors"}
    files = [_Path("/m/ideogram4_fp8_scaled.safetensors")]
    sets, pruned = matrix.artifact_sets(
        unets, files, {"ideogram4_fp8_scaled.safetensors": ["ig4_bf16.safetensors"]})
    assert {"ig4_bf16.safetensors", "ideogram4_fp8_scaled.safetensors"} == \
        {name for entry in sets for name in entry.values()}
    assert not pruned
    _sets, missed = matrix.artifact_sets(unets, files, {"nothing.safetensors": ["x.safetensors"]})
    assert missed and "nothing.safetensors" in missed[0]


def test_the_fleet_resets_after_an_infra_outcome_not_only_on_a_template_change():
    # A crashed ceremony leaves the cached mesh dirty and its procs loaded; the
    # next cell on the same template must get fresh workers, not inherit them.
    def cell(template="flux1", managed="off", gib=8.0):
        return {"template": template, "levers": {"comfy_managed": managed},
                "checkpoint_gib": gib}

    def after(observed, load="", available=100.0, template="flux1", managed="off"):
        return run.LastCell(template, managed, observed, load, available)

    assert run.needs_fleet_reset(None, cell(), 60.0) == ""
    assert run.needs_fleet_reset(after("render"), cell(), 60.0) == ""
    assert run.needs_fleet_reset(after("render"), cell(template="chroma"), 60.0) == "template"
    assert run.needs_fleet_reset(after("render"), cell(managed="on"), 60.0) == "comfy_managed"
    for outcome in ("crash", "timeout", "incomplete", "cell-error"):
        assert run.needs_fleet_reset(after(outcome), cell(), 60.0) == f"after {outcome}"
    # A refusal that arrived after the weights were on the box counts, and so
    # does one that names no load: neither cell can leave its fleet standing.
    assert run.needs_fleet_reset(after("refuse:untyped", "head held 11.34 GiB"), cell(), 60.0) \
        == "after refuse:untyped (head held 11.34 GiB)"
    assert run.needs_fleet_reset(after("refuse:untyped"), cell(), 60.0) == "after refuse:untyped"



def test_a_capacity_class_checkpoint_is_flagged_whatever_the_preset(tmp_path):
    # The stock-load preflight refuses class C for a 60 GiB-class file before
    # any sample-time guard the matrix models, and it does that at every
    # topology, so the size sets the flag and no label moves.
    assert matrix.capacity_risk_for("krea2", 60.0)
    assert not matrix.capacity_risk_for("krea2", 43.0)
    assert matrix.capacity_risk_for("wan_scail", 0.5)
    config = _rig(tmp_path)
    small, _pruned, _unmatched = matrix.build_cells(config)
    assert not [cell for cell in small if cell["capacity_risk"]]
    os.truncate(config["model_dir"] / "krea2_raw_bf16.safetensors", 60 * 2**30)
    big, _again, _rest = matrix.build_cells(config)
    grown = _pick(big, "krea2", "cfg2", artifact_quant="bf16")
    assert grown["capacity_risk"] and grown["capacity_risk_basis"] is None
    assert grown["checkpoint_gib"] >= 60.0
    assert not _pick(big, "omnigen2", "cfg2", artifact_quant="bf16")["capacity_risk"]
    # checkpoint_gib is out of ID_KEYS and out of label_cell: the flag sits
    # beside the label rather than replacing it, so a settled record resumes.
    assert [(cell["id"], cell["label"]) for cell in small] == \
        [(cell["id"], cell["label"]) for cell in big]


def test_a_capacity_flagged_cell_scores_the_wall_as_a_capacity_pass():
    cell = {"label": "refuse:untyped", "capacity_risk": True}
    assert run.verdict_for(cell, "refuse:untyped")[0] == "PASS"
    verdict, detail = run.verdict_for(cell, "refuse:C")
    assert verdict == "PASS-capacity" and "capacity boundary" in detail
    verdict, detail = run.verdict_for(cell, "refuse:K")
    assert verdict == "FINDING" and "capacity boundary" in detail
    assert run.verdict_for({"label": "refuse:untyped", "capacity_risk": False},
                           "refuse:C")[0] == "FINDING"


def test_the_measured_flux2_fp8mixed_boundary_is_not_a_generic_size_rule():
    # The boundary is one measured artifact: the 33.0 GiB file answered class C
    # ahead of cfg2's class P guard in the 2026-09 sweep. The marker records
    # that; it does not compute this host's live headroom.
    scope = {"session": "cluster", "world": 2, "auto_gate": "first_use", "preset": "cfg2"}
    assert matrix.capacity_risk_basis_for("flux2", 33.0, "fp8mixed", **scope) == \
        "flux2-fp8mixed-33gib-observed"
    assert matrix.capacity_risk_for("flux2", 33.0, "fp8mixed", **scope)
    for family, size, quant in (("flux2", 32.9, "fp8mixed"),
                                ("flux2", 33.1, "fp8mixed"),
                                ("flux2", 33.0, "bf16"),
                                ("flux", 33.0, "fp8mixed")):
        assert not matrix.capacity_risk_for(family, size, quant, **scope)
    cell = matrix._make(
        "flux2", {"klass": "image", "audio": False, "probe": False,
                  "probe_reason": "unit", "batch_node": "", "batch": 1},
        {}, "TORCH_FLASH", "cfg2", "cluster", {"image_timeout_s": 1, "world": 2},
        family="flux2", artifact_quant="fp8mixed", checkpoint_gib=33.0)
    assert cell["capacity_risk_basis"] == "flux2-fp8mixed-33gib-observed"
    # uly2 and ring2 were measured in the 2026-09-29 debt re-run.
    for preset in ("uly2", "ring2"):
        assert matrix.capacity_risk_basis_for("flux2", 33.0, "fp8mixed", **dict(scope, preset=preset)) == \
            "flux2-fp8mixed-33gib-observed"
    # No local, gate-off, different-world, or unmeasured-preset record may
    # inherit this pair's capacity answer.
    for preset, session, config, auto_gate in (
            ("single", "local", {"image_timeout_s": 1, "world": 2}, "first_use"),
            ("cfg2", "cluster", {"image_timeout_s": 1, "world": 2}, "off"),
            ("cfg2", "cluster", {"image_timeout_s": 1, "world": 4}, "first_use"),
            ("cfg2+fsdp", "cluster", {"image_timeout_s": 1, "world": 2}, "first_use")):
        control = matrix._make(
            "flux2", {"klass": "image", "audio": False, "probe": False,
                      "probe_reason": "unit", "batch_node": "", "batch": 1},
            {}, "TORCH_FLASH", preset, session, config, auto_gate=auto_gate,
            family="flux2", artifact_quant="fp8mixed", checkpoint_gib=33.0)
        assert control["capacity_risk_basis"] is None
        assert not control["capacity_risk"]


def test_the_source_digest_reads_the_runtime_not_the_harness(tmp_path):
    # A harness commit must not invalidate a settled record; a guard edit must.
    source = tmp_path / "src" / "dgx_monarch" / "nodes"
    source.mkdir(parents=True)
    (source / "loaders.py").write_text("x = 1\n")
    first = run.source_digest(tmp_path)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark" / "run.py").write_text("y = 2\n")
    assert run.source_digest(tmp_path) == first
    (source / "loaders.py").write_text("x = 2\n")
    edited = run.source_digest(tmp_path)
    assert edited != first
    (source / "loaders.py").rename(source / "store_load.py")
    assert run.source_digest(tmp_path) not in (first, edited)


def _stub_nccl(monkeypatch, loaded, built=(2, 29, 7), status=0):
    """torch reports `built`; the library handle answers `loaded` (None: no library) and returns `status`."""
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(nccl=types.SimpleNamespace(
        version=lambda: built)))
    monkeypatch.setitem(sys.modules, "torch", fake)

    def open_library(name):
        if loaded is None:
            raise OSError(f"{name}: cannot open shared object file")

        def get_version(ref):
            ref._obj.value = loaded
            return status
        return types.SimpleNamespace(ncclGetVersion=get_version)
    monkeypatch.setattr(run.ctypes, "CDLL", open_library)


def test_the_nccl_pin_names_the_loaded_library_not_the_build(monkeypatch):
    _stub_nccl(monkeypatch, 23007)
    assert run._nccl_version() == ("2.30.7", "loaded-library")


@pytest.mark.parametrize(("code", "text"), [(22809, "2.28.9"), (23203, "2.32.3"),
                                            (21901, "2.19.1"), (30010, "3.0.10")])
def test_the_nccl_pin_formats_any_version_code(monkeypatch, code, text):
    _stub_nccl(monkeypatch, code)
    assert run._nccl_version()[0] == text


def test_the_nccl_pin_falls_back_to_the_build_when_the_library_will_not_answer(monkeypatch):
    _stub_nccl(monkeypatch, None)
    assert run._nccl_version() == ("2.29.7", "torch-runtime")
    _stub_nccl(monkeypatch, 0)
    assert run._nccl_version() == ("2.29.7", "torch-runtime")
    _stub_nccl(monkeypatch, 23007, status=1)
    assert run._nccl_version() == ("2.29.7", "torch-runtime")
    _stub_nccl(monkeypatch, 23007)
    monkeypatch.setattr(run.ctypes, "CDLL", lambda name: object())
    assert run._nccl_version() == ("2.29.7", "torch-runtime")


def test_the_nccl_pin_falls_back_to_the_harness_env_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(run, "_package", lambda name: "2.1.0")
    assert run._nccl_version() == ("2.1.0", "harness-env")


def test_the_session_pin_carries_the_loaded_nccl_and_a_swap_blocks_the_resume(monkeypatch, tmp_path):
    _stub_nccl(monkeypatch, 23007)
    monkeypatch.setattr(run, "get", lambda *args, **kwargs: {"system": {"pytorch_version": "2.12.0"}})
    config = {"driver": "http://driver", "comfy_dir": tmp_path, "world": 2}
    pin = run.session_pin(config)
    assert (pin["nccl"], pin["nccl_source"]) == ("2.30.7", "loaded-library")
    _stub_nccl(monkeypatch, 23203)
    assert run.session_pin(config)["nccl"] == "2.32.3"
    reference = {"kind": "cell", "cell": "abc123", "bar": "step-nrms", "probe_steps": 1}
    done = {"pin": pin, "graph_digest": "g", "verdict": "PASS", "observed": "refuse:P",
            "cell": {"label": "refuse:P", "reference": dict(reference)}}
    assert run.resumable(done, pin, "g", "refuse:P", reference)
    assert not run.resumable(done, run.session_pin(config), "g", "refuse:P", reference)


def test_the_resume_identity_skips_the_provenance_the_pin_carries():
    pin = {"comfy_commit": "c1", "monarch_commit": "m1", "source_digest": "s1",
           "torch": "2.9.0", "torch_source": "driver", "nccl": "2.28.3",
           "nccl_source": "torch-runtime", "gate_protocol_version": "12",
           "driver": "http://a-driver:8191", "world": 2}
    reference = {"kind": "cell", "cell": "abc123", "bar": "step-nrms", "probe_steps": 1}
    done = {"pin": dict(pin), "graph_digest": "g1", "verdict": "PASS",
            "observed": "refuse:P",
            "cell": {"label": "refuse:P", "reference": dict(reference)}}
    assert run.resumable(done, pin, "g1", "refuse:P", reference)
    assert run.resumable(done, dict(pin, monarch_commit="m2", torch_source="harness-env",
                                    nccl_source="harness-env",
                                    driver="http://another-driver:8191"), "g1", "refuse:P",
                         reference)
    for key in run.RESUME_KEYS:
        assert not run.resumable(done, dict(pin, **{key: "moved"}), "g1", "refuse:P", reference)
    assert not run.resumable(done, pin, "g2", "refuse:P", reference)
    assert not run.resumable({**done, "observed": "crash"}, pin, "g1", "refuse:P", reference)
    assert not run.resumable({}, pin, "g1", "refuse:P", reference)
    # A settled record whose label the matrix has since moved was scored
    # against the old label, so the cell runs again.
    assert not run.resumable(done, pin, "g1", "refuse:untyped", reference)
    # A new capacity basis can rescore an existing class-C record from a
    # finding to PASS-capacity, so it likewise cannot resume an older record.
    boundary = "flux2-fp8mixed-33gib-observed"
    assert not run.resumable(done, pin, "g1", "refuse:P", reference, boundary)
    risk_done = {**done, "cell": {**done["cell"], "capacity_risk_basis": boundary}}
    assert run.resumable(risk_done, pin, "g1", "refuse:P", reference, boundary)
    assert not run.resumable({k: v for k, v in done.items() if k != "cell"},
                             pin, "g1", "refuse:P", reference)
    # Every record written before the source digest existed re-runs once.
    assert not run.resumable({**done, "pin": {k: v for k, v in pin.items()
                                              if k != "source_digest"}}, pin, "g1",
                             "refuse:P", reference)


def test_a_label_wave_leaves_the_waivable_cells_to_the_waiver_wave(tmp_path):
    cells, _pruned, _unmatched = matrix.build_cells(_rig(tmp_path))
    waivable = [cell for cell in cells if cell["session"] == "cluster"
                and not cell["sampled_out"] and cell["label"] == "refuse:K"
                and (cell["waiver"] or cell["waiver_guards"])]
    assert waivable
    # A limit keeps the one-video-per-run rule out of the comparison.
    chosen, pruned = run.select(
        cells, _Args(session="cluster", cells="refuse:K", limit=len(cells)))
    assert set(pruned["waivers_excluded"]) == {cell["id"] for cell in waivable}
    assert not {cell["id"] for cell in chosen} & {cell["id"] for cell in waivable}
    kept, _cut = run.select(cells, _Args(session="cluster", cells="refuse:K",
                                         limit=len(cells), include_waivers=True))
    assert {cell["id"] for cell in kept} >= {cell["id"] for cell in waivable}
    # An id or a template names the cell itself, so no wave drops it silently.
    named, _cut = run.select(cells, _Args(session="cluster", cells=waivable[0]["id"],
                                          limit=len(cells)))
    assert [cell["id"] for cell in named] == [waivable[0]["id"]]


def test_a_cell_that_loaded_weights_leaves_evidence_in_its_memory_sample():
    # Peak against the minimum, never against the first row: a fleet still
    # draining the previous load delivers its residue in that first row.
    def cold(**over):
        leg = {"outcome": "refuse:P", "wall_s": 4.0,
               "memory": {"head": {"AnonPages": {"min": 10.0, "peak": 10.44, "last": 10.4},
                                   "MemAvailable": {"min": 90.0, "peak": 92.0, "last": 91.0}}}}
        leg.update(over)
        return {"legs": {"cold": leg}}

    assert run.load_evidence(cold()) == ""
    assert run.load_evidence({"legs": {}}) == ""
    grew = cold(memory={"head": {"AnonPages": {"min": 9.3, "peak": 26.6, "last": 9.6}}})
    assert "17.3 GiB" in run.load_evidence(grew)
    # A slab load is mmap and file-backed, so it moves no anonymous memory at
    # all and the wall is the only signal that sees it.
    slab = cold(wall_s=42.4,
                memory={"head": {"AnonPages": {"min": 23.1, "peak": 24.04, "last": 23.9}}})
    assert run.load_evidence(slab) == "a 42.4 s cold leg"
    assert run.load_evidence(cold(memory={"memory_error": "the sampler thread did not stop"}))
    assert run.load_evidence(cold(memory={"head": {"read_error": "rc 255: transport lost"}}))
    # A guard that refuses in two seconds and then renders under its card has
    # loaded the checkpoint just the same, so the waived leg is read too.
    quiet = cold(outcome="refuse:K", wall_s=2.0)["legs"]["cold"]
    assert run.load_evidence({"legs": {"cold": quiet}}) == ""
    assert run.load_evidence({"legs": {
        "cold": quiet,
        "waived": {"outcome": "render", "wall_s": 121.0, "memory": {}}}}) \
        == "a 121.0 s waived leg"


def test_only_a_dirty_mesh_crash_earns_the_one_retry(tmp_path):
    def crashed(kind: str, outcome: str = "crash") -> dict:
        return {"legs": {"cold": {"outcome": outcome, "exception_type": kind}}}

    assert run.recoverable_crash(crashed("dgx_monarch.mesh.MeshAttachError"))
    assert run.recoverable_crash(
        crashed("dgx_monarch.mesh_setup_state.SampleResultBusyError"))
    # The type is joined across status messages, so every line is read.
    assert run.recoverable_crash(crashed("ValueError\ndgx_monarch.mesh.MeshAttachError"))
    assert run.recoverable_crash(crashed("dgx_monarch.nodes.gate_fsdp.FsdpGateProofError"))
    assert not run.recoverable_crash(crashed("dgx_monarch.mesh.MeshAttachError", "refuse:P"))
    assert not run.recoverable_crash({"legs": {}})
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "abc.json").write_text("{}")
    assert run.keep_attempt(tmp_path, "abc").name == "abc.attempt1.json"
    (cells / "abc.json").write_text("{}")
    assert run.keep_attempt(tmp_path, "abc").name == "abc.attempt2.json"


def test_a_floor_that_stops_moving_returns_early_instead_of_expiring(monkeypatch):
    assert not run.floor_stalled([57.4] * (run.FLOOR_STALL_POLLS - 1))
    assert run.floor_stalled([57.4] * run.FLOOR_STALL_POLLS)
    assert not run.floor_stalled([57.4] * run.FLOOR_STALL_POLLS + [52.5])
    assert not run.floor_stalled([float(step) for step in range(run.FLOOR_STALL_POLLS * 2)])
    kilobytes = int(57.4 * 2**20)
    monkeypatch.setattr(run, "_on", lambda host, command, timeout:
                        (f"MemAvailable: {kilobytes} kB\n", ""))
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    floor = run.wait_for_memory([""], 60.0, 600.0)
    assert floor["cleared"] is False and floor["stalled"] == run.FLOOR_STALL_POLLS


def test_a_rejection_that_names_a_template_node_condemns_that_template_only():
    def body(nodes: dict) -> str:
        return "400 Bad Request: " + json.dumps(
            {"error": {"type": "prompt_outputs_failed_validation"}, "node_errors": nodes})

    assert run.rejection_scope(body({"12": {"class_type": "LTXVEmptyLatentAudio",
                                            "errors": [{"type": "required_input_missing"}]}})) \
        == "template"
    # No node at all, or the Init node every template carries: nothing would run.
    assert run.rejection_scope(body({})) == "session"
    assert run.rejection_scope(body({"1": {"class_type": matrix.INIT_CLASS}})) == "session"
    assert run.rejection_scope(body({"1": {"class_type": matrix.INIT_CLASS},
                                     "12": {"class_type": "LTXVEmptyLatentAudio"}})) == "session"
    assert run.rejection_scope("400 Bad Request: " + json.dumps({"error": {"type": "x"}})) \
        == "unknown"
    assert run.rejection_scope('400 Bad Request: {"node_errors": {"12": {"class_') == "unknown"
    assert run.rejection_scope("") == "unknown"


def _prompt_template(config: dict, name: str, family: str, unet: str,
                     positive: str, negative: str, cfg: float = 3.5) -> None:
    """A template on a real signature checkpoint, carrying a prompt pair."""
    _checkpoint(config["model_dir"], unet, family)
    graph = _graph(unet, cfg=cfg)
    graph["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive}}
    graph["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative}}
    graph["4"]["inputs"].update({"positive": ["6", 0], "negative": ["7", 0]})
    (config["out_dir"] / "graphs" / f"{name}.json").write_text(json.dumps(graph))


def _characters(_family: str, text: str) -> int:
    """A counter the test drives: one token per character."""
    return len(text)


def test_the_families_that_shard_text_apart_are_the_flux_family_adapters():
    # Together the two sets are the adapters whose forwards shard text apart
    # from image, so a fifth one landing upstream is a test failure rather than
    # a silent gap. Which set a family lands in is its own probe attribute: a
    # family that grows one moves without a name being edited here.
    from dgx_monarch.adapters import ADAPTERS

    flux_family = {adapter.family for adapter in ADAPTERS
                   if type(adapter).__module__.endswith("flux_family")}
    assert matrix.SP_EXACT_TEXT_FAMILIES | matrix.SP_PAD_VOUCHED_FAMILIES == flux_family
    assert not matrix.SP_EXACT_TEXT_FAMILIES & matrix.SP_PAD_VOUCHED_FAMILIES
    # Every shipped member is vouched since 2026-09-09, so the exact-text set is
    # empty. It stays in the rule for the next family that arrives without
    # evidence, and the next assertion fails when one does.
    assert matrix.SP_EXACT_TEXT_FAMILIES == set()
    assert matrix.SP_PAD_VOUCHED_FAMILIES == {"chroma", "flux", "flux2", "longcat"}
    by_family = {adapter.family: adapter for adapter in ADAPTERS}
    assert all(by_family[family].usp_pad_exclusion_probe is not None
               for family in matrix.SP_PAD_VOUCHED_FAMILIES)
    assert all(by_family[family].usp_pad_exclusion_probe is None
               for family in matrix.SP_EXACT_TEXT_FAMILIES)


def test_a_prompt_token_count_decides_the_sequence_parallel_label(
    tmp_path, monkeypatch
):
    # The graph carries the prompt, not its encoded length, so a uly2 cell on a
    # family that shards text apart from image is labelled from the count. 101
    # tokens do not divide 2. Every shipped flux-family member names a
    # pad-exclusion probe since 2026-09-09, so the test marks flux unvouched to
    # keep this rule covered: it is what the next member arriving without
    # evidence will meet.
    monkeypatch.setattr(matrix, "SP_EXACT_TEXT_FAMILIES", frozenset({"flux"}))
    monkeypatch.setattr(matrix, "SP_PAD_VOUCHED_FAMILIES", frozenset({"chroma"}))
    config = _rig(tmp_path)
    config.update(toggles={"auto_gate": ["off"]},
                  toggle_presets={"auto_gate": ["uly2+fsdp"]})
    _prompt_template(config, "fluxodd", "flux", "flux_bf16.safetensors",
                     "x" * 101, "y" * 28)
    _prompt_template(config, "fluxeven", "flux", "flux_bf16.safetensors",
                     "x" * 44, "y" * 44)
    cells, _pruned, _unmatched = matrix.build_cells(config, count_tokens=_characters)
    odd = _pick(cells, "fluxodd", "uly2")
    assert (odd["label"], odd["basis"]) == ("refuse:P", "token-count")
    assert odd["text_tokens"] == {"positive": 101, "negative": 28}
    # The contract is typed under FSDP too: since 2026-09-03 the clean-reload
    # proof settles a typed refusal, so the first_use cell answers the class its
    # resident leg answers. A proof that aborts on the refusal instead leaves
    # the retry denied untagged by the process-local verdict.
    assert _pick(cells, "fluxodd", "uly2+fsdp",
                 auto_gate="first_use")["label"] == "refuse:P"
    assert _pick(cells, "fluxodd", "uly2+fsdp", auto_gate="off")["label"] == "refuse:P"
    assert _pick(cells, "fluxeven", "uly2+fsdp", auto_gate="off")["label"] == "render"
    assert _pick(cells, "fluxodd", "ring2")["label"] == "refuse:P"
    even = _pick(cells, "fluxeven", "uly2")
    assert (even["label"], even["basis"]) == ("render", "token-count")
    # cfg2 pads and masks on this family, so the count decides nothing there.
    assert _pick(cells, "fluxodd", "cfg2")["label"] == "render"


@pytest.mark.parametrize("family,unet", [
    ("chroma", "chroma_bf16.safetensors"),
    ("flux", "flux_bf16.safetensors"),
])
def test_a_vouched_family_renders_an_odd_count_and_takes_the_ring_card(
    tmp_path, family, unet
):
    # Every flux-family member excludes its divisibility pad rows, so an odd
    # count renders on pure ulysses. Ring has no full-sequence point to exclude
    # them at, so it takes the waivable ring_pad card instead of the class P
    # shard-time refusal.
    config = _rig(tmp_path)
    config.update(toggles={"auto_gate": ["off"]},
                  toggle_presets={"auto_gate": ["uly2+fsdp"]})
    _prompt_template(config, "podd", family, unet, "x" * 101, "y" * 28)
    _prompt_template(config, "peven", family, unet, "x" * 44, "y" * 44)
    cells, _pruned, _unmatched = matrix.build_cells(config, count_tokens=_characters)
    odd = _pick(cells, "podd", "uly2")
    assert (odd["label"], odd["basis"]) == ("render", "token-count")
    assert odd["text_tokens"] == {"positive": 101, "negative": 28}
    assert _pick(cells, "podd", "uly2+fsdp", auto_gate="off")["label"] == "render"
    ring = _pick(cells, "podd", "ring2")
    assert (ring["label"], ring["basis"]) == ("refuse:K", "token-count")
    assert ring["waiver"] and "ring_pad" in ring["waiver_guards"]
    # An even count carries no pad row at all, so ring keeps its render.
    assert _pick(cells, "peven", "ring2")["label"] == "render"
    assert _pick(cells, "peven", "uly2")["label"] == "render"


def test_a_measured_family_takes_the_nvfp4_card_on_every_sharded_topology(tmp_path):
    # The shared activation scale is exact and chroma nvfp4 still reads past
    # the fidelity floor with it on (2026-09-05), so the bar is the family's
    # measurement and not the hook's coverage: every topology the scale reduces
    # across takes the waivable class K card, and the waived leg is what
    # measures the cell.
    config = _rig(tmp_path)
    _prompt_template(config, "chroma", "chroma", "chroma_bf16.safetensors",
                     "x" * 44, "y" * 44)
    _checkpoint(config["model_dir"], "chroma_nvfp4.safetensors", "chroma")
    cells, _pruned, _unmatched = matrix.build_cells(config, count_tokens=_characters)
    for preset in ("auto", "uly2", "cfg2", "ring2"):
        cell = _pick(cells, "chroma", preset, artifact_quant="nvfp4")
        assert (cell["label"], cell["waiver"]) == ("refuse:K", True), preset
        assert "shard_quant_scale:chroma" in cell["waiver_guards"], preset
    # One GPU sees the whole tensor, and the same topologies on a bf16 artifact
    # carry no nvfp4 layer to quantize.
    single = [cell for cell in cells if cell["template"] == "chroma"
              and cell["session"] == "local" and "nvfp4" in cell["artifact_quant"]]
    assert single and {cell["label"] for cell in single} == {"render"}
    assert _pick(cells, "chroma", "uly2", artifact_quant="bf16")["label"] == "render"
    # The FSDP launch contract answers before this bar, on the artifact name.
    assert _pick(cells, "chroma", "uly2+fsdp",
                 artifact_quant="nvfp4")["label"] == "refuse:P"
    # A family not measured past the floor is refused nothing: krea2 nvfp4 reads
    # under it on uly2.
    assert _pick(cells, "krea2", "uly2", artifact_quant="nvfp4")["label"] == "render"


def test_an_unequal_prompt_pair_no_longer_decides_the_cfg_label(tmp_path):
    # On two prompts of different length, a family that pads no conditioning
    # makes two model calls of batch 1 and the slice has nothing to cut. The
    # per-cond dispatch catches that pair on every family since 2026-09-05, so
    # the cell renders and the token counts decide nothing here. The label is
    # unmeasured on anima: the dispatch was measured on boogu and omnigen2.
    config = _rig(tmp_path)
    config.update(toggles={"auto_gate": ["off"]},
                  toggle_presets={"auto_gate": ["cfg2+fsdp"]})
    _prompt_template(config, "animaragged", "anima", "anima_bf16.safetensors",
                     "x" * 102, "y" * 36)
    _prompt_template(config, "animalevel", "anima", "anima_bf16.safetensors",
                     "x" * 102, "x" * 102)
    cells, _pruned, _unmatched = matrix.build_cells(config, count_tokens=_characters)
    ragged = _pick(cells, "animaragged", "cfg2")
    assert (ragged["label"], ragged["basis"]) == ("render", "cond-fold-unknown")
    # No constant, so two texts that differ predict neither path.
    assert ragged["cfg_path"] == "either"
    for gate in ("first_use", "off"):
        assert _pick(cells, "animaragged", "cfg2+fsdp", auto_gate=gate)["label"] == "render"
    level = _pick(cells, "animalevel", "cfg2")
    assert level["label"] == "render" and level["cfg_path"] == "slice"


def test_an_unequal_caption_renders_on_every_kernel_since_the_dispatch(tmp_path):
    # boogu offsets every image token's rope by the caption length, so a padded
    # caption shifts the rope against a single-GPU render. A family that declares
    # a per-cond constant is never cond-padded (since 2026-09-04): the pad could
    # never fold its constant and would hand one rank alone a cond its forward
    # refuses. With no pad the rope defect cannot fire, so all three kernels
    # render through the per-cond dispatch.
    config = _rig(tmp_path)
    config.update(image_presets=["auto", "cfg2", "cfg2+fsdp"], video_presets=["auto"],
                  image_attention=["TORCH_FLASH", "TORCH_CUDNN", "SAGE_AUTO"])
    _prompt_template(config, "boogupair", "boogu", "boogu_bf16.safetensors",
                     "a caption of its own", "")
    _prompt_template(config, "boogulevel", "boogu", "boogu_bf16.safetensors",
                     "one caption", "one caption")
    cells, _pruned, _unmatched = matrix.build_cells(config)
    for preset in ("cfg2", "cfg2+fsdp"):
        for kernel in ("TORCH_CUDNN", "SAGE_AUTO", "TORCH_FLASH"):
            unequal = _pick(cells, "boogupair", preset, attention=kernel)
            assert (unequal["label"], unequal["basis"]) == ("render", "per-cond-dispatch")
            assert unequal["cfg_path"] == "dispatch"
        # Two prompts of one length fold, and the slice is faster there.
        level = _pick(cells, "boogulevel", preset, attention="SAGE_AUTO")
        assert level["label"] == "render" and level["cfg_path"] == "slice"
    # A sequence topology runs neither path.
    assert _pick(cells, "boogupair", "auto", attention="SAGE_AUTO")["label"] == "render"
    assert _pick(cells, "boogupair", "auto", attention="SAGE_AUTO")["cfg_path"] == "none"


def test_the_constant_families_render_an_unequal_pair_by_dispatch(tmp_path):
    # boogu and omnigen2 publish the real caption length as a CONDConstant, so
    # two prompts of different length never enter one batched call. The
    # per-cond dispatch needs none: each rank runs one conditioning at the
    # sampler seam. The evidence is the two prompt texts, which is all the
    # matrix has without a tokenizer for these streams.
    config = _rig(tmp_path)
    config.update(image_presets=["auto", "cfg2"], video_presets=["auto"],
                  image_attention=["TORCH_FLASH"])
    _prompt_template(config, "omnipair", "omnigen2", "omnigen2_bf16.safetensors",
                     "a long and particular caption", "blurry")
    _prompt_template(config, "omnilevel", "omnigen2", "omnigen2_bf16.safetensors",
                     "one caption here", "one caption here")
    cells, _pruned, _unmatched = matrix.build_cells(config)
    unequal = _pick(cells, "omnipair", "cfg2")
    assert (unequal["label"], unequal["basis"]) == ("render", "per-cond-dispatch")
    assert unequal["cfg_path"] == "dispatch"
    # Two prompts of one length still fold, and the slice is faster there.
    equal = _pick(cells, "omnilevel", "cfg2")
    assert equal["label"] == "render" and equal["cfg_path"] == "slice"
    # A cell below cfg 2 claims neither path.
    assert _pick(cells, "omnipair", "auto")["cfg_path"] == "none"


def test_a_family_with_no_constant_predicts_neither_cfg_path(tmp_path):
    # The prompt-text reading only answers where a per-cond CONDConstant is the
    # caption length. A family that publishes none leaves the pair's fate on a
    # conditioning shape no tokenizer here can read, and comfy's own concat rule
    # folds two cross-attention conds of unequal length when the lowest common
    # multiple is close enough, so two texts that differ prove nothing. The cell
    # renders through the dispatch or the slice and the basis says the path was
    # unknowable. The ernie hardware leg is NOT RUN (2026-09-05 record).
    config = _rig(tmp_path)
    config.update(image_presets=["auto", "cfg2"], video_presets=["auto"],
                  image_attention=["TORCH_FLASH"])
    _prompt_template(config, "erniepair", "ernie", "ernie_bf16.safetensors",
                     "a caption of its own", "a much shorter one")
    _prompt_template(config, "ernielevel", "ernie", "ernie_bf16.safetensors",
                     "one caption", "one caption")
    cells, _pruned, _unmatched = matrix.build_cells(config)
    unread = _pick(cells, "erniepair", "cfg2")
    assert (unread["label"], unread["basis"]) == ("render", "cond-fold-unknown")
    assert unread["cfg_path"] == "either"
    # One text is one shape on any family, so an equal pair still folds.
    level = _pick(cells, "ernielevel", "cfg2")
    assert level["label"] == "render" and level["cfg_path"] == "slice"


def test_a_count_nobody_can_read_keeps_the_label_and_says_so(tmp_path):
    # An unreadable tokenizer leaves the label the other rules give and names
    # the basis, so the report never reads a guessed count as evidence.
    config = _rig(tmp_path)
    _prompt_template(config, "fluxodd", "flux", "flux_bf16.safetensors",
                     "x" * 101, "y" * 28)
    cells, _pruned, _unmatched = matrix.build_cells(
        config, count_tokens=lambda family, text: None)
    unread = _pick(cells, "fluxodd", "uly2")
    assert (unread["label"], unread["basis"]) == ("render", "token-count-unavailable")
    assert unread["text_tokens"] == {}
    # One side alone is no pair: counting it twice would say two streams match.
    assert matrix.text_token_counts({"positive": "a", "negative": "b"}, "chroma",
                                    lambda family, text: 4 if text == "a" else None) == {}


# The chroma positive prompt the runtime counted at 101 text tokens when it
# refused that render at uly2. The shipped prompt was evened on 2026-09-02, so
# the calibration lives here rather than in a file that moves.
CHROMA_101_TOKENS = (
    "This is a nature documentary close-up photograph of the right side of "
    "the face of a tiger. The photograph is centered on it's highly detailed "
    "and speckled eye surrounded by intricately detailed fur. Overlaid at the "
    "center of the image is a title text that says \"CHROMA1-HD\" in a large "
    "white 3D letters. Amateur photography. Unfiltered. Real life. Natural "
    "light. Subtle shadows. "
)


class _StubEncoding:
    def __init__(self, ids) -> None:
        self.ids = ids


class _StubTokenizer:
    """One id a character, which is enough to prove the floor and the streams."""

    def encode(self, text: str) -> _StubEncoding:
        return _StubEncoding(list(text))


def test_the_counter_pads_to_the_floor_and_counts_nothing_it_cannot_read(monkeypatch):
    # The calibration below needs comfy's tokenizer file and skips without one.
    # This one runs everywhere: the floor, a family with no shipped tokenizer
    # file, and the weight syntax comfy would split into segments.
    counter = matrix.TokenCounter(Path("/no/such/directory"))
    monkeypatch.setattr(matrix.TokenCounter, "_tokenizer",
                        lambda self, name: _StubTokenizer())
    assert counter("chroma", "x" * 44) == 44
    # comfy pads the flux stream to 256 before the adapter shards it.
    assert counter("flux", "a tiger") == 256
    assert counter("cogvideo", "a tiger") == 226
    # An empty prompt still carries the end token comfy appends.
    assert counter("chroma", "") == 1
    assert counter("wan", "a tiger") is None
    assert counter("chroma", "(a tiger:1.2)") is None
    assert counter("chroma", "embedding:tiger") is None


def test_the_counter_reproduces_the_length_the_runtime_named():
    # A counter that misses this number is not counting what comfy hands the
    # chroma adapter, and every label it feeds would be a guess. Rig-only: it
    # reads the tokenizer file ComfyUI ships, so a box without one skips.
    directory = matrix.tokenizer_dir({"comfy_dir": Path(matrix.DEFAULTS["comfy_dir"]).expanduser()})
    if not (directory / matrix.T5_TOKENIZER).is_file():
        pytest.skip("no comfy tokenizer file on this box")
    counter = matrix.TokenCounter(directory)
    assert counter("chroma", CHROMA_101_TOKENS) == 101
    # An empty prompt still carries the end token comfy appends: chroma's blank
    # negative is one token.
    assert counter("chroma", "") == 1
    # A family with no shipped tokenizer file, and a prompt comfy would split
    # into weighted segments, are both left uncounted.
    assert counter("wan", CHROMA_101_TOKENS) is None
    assert counter("chroma", "(a tiger:1.2)") is None
    # comfy pads the flux stream to 256 before the adapter shards it, so the
    # floor is the length that decides, not the prompt.
    assert counter("flux", "a tiger") == 256


def test_a_conditioning_node_does_not_hand_both_slots_one_prompt():
    # A guide or image-conditioning node returns each stream on the slot it
    # took it from. Following any input instead reads the positive prompt twice
    # and calls a ragged pair level.
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
        "3": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["1", 0], "negative": ["2", 0], "frame_rate": 25.0}},
        "4": {"class_type": "DGXMonarchCFGGuider",
              "inputs": {"positive": ["3", 0], "negative": ["3", 1], "cfg": 3.0}},
    }
    assert matrix.prompt_texts(graph) == {"positive": "positive", "negative": "negative"}
    # A slot the walk cannot resolve is no pair at all.
    assert matrix.prompt_texts({"4": graph["4"]}) == {}
    assert matrix.prompt_texts({"1": graph["1"]}) == {}


def test_block_scaled_tokens_under_fsdp_split_uniform_from_mixed(tmp_path):
    # A uniform block-scaled file names its own kind to the live detector and
    # the launch contract refuses it with the class P card (typed since
    # 2026-09-09; the comfy-bump validation of 2026-09-29 observed refuse:P for
    # both mxfp8 and nvfp4); a mixed one names the admitted fp8, reaches the
    # shard-layout check and gets its own typed card.
    config = _rig(tmp_path)
    fp8 = torch.float8_e4m3fn
    _checkpoint(config["model_dir"], "krea2_raw_mxfp8.safetensors", "krea2", fp8)
    _checkpoint(config["model_dir"], "krea2_raw_nvfp4_mixed.safetensors", "krea2", fp8)
    cells, _pruned, _unmatched = matrix.build_cells(config)
    nvfp4 = _pick(cells, "krea2", "uly2+fsdp", artifact_quant="nvfp4")
    mxfp8 = _pick(cells, "krea2", "uly2+fsdp", artifact_quant="mxfp8")
    mixed = _pick(cells, "krea2", "uly2+fsdp", artifact_quant="nvfp4_mixed")
    assert (nvfp4["label"], nvfp4["basis"]) == ("refuse:P", "name-token")
    assert (mxfp8["label"], mxfp8["basis"]) == ("refuse:P", "name-token")
    assert (mixed["label"], mixed["basis"]) == ("refuse:P", "name-token")
    # The walk is ordered: a substring test would hand nvfp4_mixed the nvfp4
    # entry, which names a different refusal site.
    assert next(iter(matrix.BLOCK_SCALED)) == "nvfp4_mixed"
    # No token moves a resident cell.
    assert _pick(cells, "krea2", "uly2", artifact_quant="mxfp8")["label"] == "render"
    assert _pick(cells, "krea2", "uly2", artifact_quant="nvfp4_mixed")["label"] == "render"


def test_an_upstream_gated_template_is_skipped_and_named(tmp_path):
    # comfy admits the artifact and the adapter cannot run it, so a reference
    # render thrashes the host instead of refusing (2026-09-02).
    config = _rig(tmp_path)
    config["upstream_gated"] = ["krea2"]
    cells, pruned, _unmatched = matrix.build_cells(config)
    gated = [cell for cell in cells if cell["template"] == "krea2"]
    assert gated and {cell["label"] for cell in gated} == {"skip:upstream-gated"}
    assert {cell["basis"] for cell in gated} == {"config"}
    assert any("krea2: upstream-gated by the sweep config" in line for line in pruned)
    # The counts table counts them the way it counts every other skip.
    assert "| skip:upstream-gated |" in matrix.counts_table(cells)
    chosen, cut = run.select(cells, _Args(session="cluster"))
    assert not [cell for cell in chosen if cell["template"] == "krea2"]
    assert {cell["id"] for cell in gated} & set(cut["skipped"])


def test_an_l2p_header_refuses_before_the_static_template_skip(tmp_path):
    """The sweep asks the runtime header discriminator, not a family name."""
    config = _rig(tmp_path)
    models, graphs = config["model_dir"], config["out_dir"] / "graphs"
    # An older config's upstream_gated entry must not hide a real loader
    # refusal. Its template name says DCT, but the runtime decides from the
    # selected file.
    config["upstream_gated"] = ["zimage-dct"]
    _checkpoint(models, "zimage_dct_l2p.safetensors", "zimage")
    l2p_path = models / "zimage_dct_l2p.safetensors"
    save_file({key: torch.zeros(2, 2, dtype=torch.bfloat16) for key in
               [*FAMILY_KEYS["zimage"], "local_decoder.blocks.0.weight"]}, str(l2p_path))
    _checkpoint(models, "zimage_dct_control.safetensors", "zimage")
    dct_path = models / "zimage_dct_control.safetensors"
    save_file({key: torch.zeros(2, 2, dtype=torch.bfloat16) for key in
               [*FAMILY_KEYS["zimage"], "dec_net.blocks.0.weight"]}, str(dct_path))
    _checkpoint(models, "zimage_latent.safetensors", "zimage")
    for template, unet in (("zimage-dct", l2p_path.name),
                           ("zimage-dct-control", dct_path.name),
                           ("zimage-latent", "zimage_latent.safetensors")):
        (graphs / f"{template}.json").write_text(json.dumps(_graph(unet)))

    cells, _pruned, _unmatched = matrix.build_cells(config)
    l2p = _pick(cells, "zimage-dct", "cfg2")
    dct = _pick(cells, "zimage-dct-control", "cfg2")
    latent = _pick(cells, "zimage-latent", "cfg2")
    assert (l2p["label"], l2p["basis"]) == ("refuse:P", "guard")
    # The near-neighbour controls share the family word but do not carry the
    # local_decoder-without-dec_net layout the shipped guard refuses.
    assert dct["label"] == latent["label"] == "render"


TEMPLATE_DIR = REPO / "example_workflows"


@contextlib.contextmanager
def _generator_widget_names():
    """The generator's widget map, with the generator's own stubs in place.

    A sampler node reads its dropdown out of `comfy.samplers` and a loader
    node reads its file list out of `folder_paths`. The generator installs
    both around each of its own widget reads and puts them back; this holds
    them for the whole conversion and restores exactly what was there. The
    snapshot below is taken after the import, which since 2026-10-05 leaves no
    stub in `sys.modules`, so it records the caller's state and not the stubs. A
    sweep runs in its own process and meets none of this.
    """
    names = convert.repo_widget_names(REPO)
    generator = sys.modules["gen_templates"]
    stubs = {"comfy": generator.comfy, "comfy.samplers": generator.samplers_mod,
             "folder_paths": generator.fp}
    before = {name: sys.modules[name] for name in stubs if name in sys.modules}
    sys.modules.update(stubs)
    try:
        yield names
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
        sys.modules.update(before)


def _template(stem: str) -> dict:
    """One shipped template, as the API graph the sweep runs."""
    path = TEMPLATE_DIR / f"{stem}.json"
    if not path.is_file():
        path = REPO / "tests" / "fixtures" / "workflows" / "generated" / f"{stem}.json"
    with _generator_widget_names() as names:
        return convert.convert(json.loads(path.read_text()), names)


def test_the_probe_rewrite_reaches_a_scheduler_node_and_a_sigma_list():
    # Neither graph keeps its step count in a KSampler widget. Read from those
    # widgets alone, both lose their probe leg and fall back to two full warm
    # renders, which the one-step floor cannot score.
    ideogram4 = _template("dgx-monarch-ideogram4-t2i")
    assert convert.probe_support(ideogram4) == (True, "")
    convert.set_probe_steps(ideogram4, 1)
    assert [node["inputs"]["steps"] for node in ideogram4.values()
            if node["class_type"] == "Ideogram4Scheduler"] == [1]

    ltx = _template("dgx-monarch-ltx-t2v")
    assert convert.probe_support(ltx) == (True, "")
    full = [node["inputs"]["sigmas"] for node in ltx.values()
            if node["class_type"] == "ManualSigmas"]
    assert len(full) == 1 and full[0].count(",") == 8
    convert.set_probe_steps(ltx, 1)
    # One interval, written in the template's own first and last value: a
    # candidate and its reference have to be handed the identical schedule.
    assert [node["inputs"]["sigmas"] for node in ltx.values()
            if node["class_type"] == "ManualSigmas"] == ["1.0, 0.0"]
    assert convert.probe_sigmas("1.0, 0.75, 0.5, 0.25, 0.0", 2) == "1.0, 0.5, 0.0"
    # A schedule already at or under the probe length is left alone.
    assert convert.probe_sigmas("1.0, 0.0", 1) == "1.0, 0.0"


def test_a_staged_graph_takes_one_step_in_every_stage():
    # One step over the whole schedule leaves stage two an empty window, so each
    # stage takes one step of its own.
    for stem, tail in (("dgx-monarch-wan22-t2v", 20), ("dgx-monarch-wan-bernini", 10000)):
        graph = _template(stem)
        stages = [node["inputs"] for node in graph.values()
                  if node["class_type"] == "DGXMonarchKSamplerAdvanced"]
        assert len(stages) == 2 and stages[1]["end_at_step"] == tail
        assert convert.probe_support(graph) == (True, "")
        convert.set_probe_steps(graph, 1)
        cut = sorted((node["inputs"] for node in graph.values()
                      if node["class_type"] == "DGXMonarchKSamplerAdvanced"),
                     key=lambda inputs: inputs["start_at_step"])
        assert [(inputs["steps"], inputs["start_at_step"], inputs["end_at_step"])
                for inputs in cut] == [(2, 0, 1), (2, 1, 2)]
    # A two-stage graph that schedules each stage apart keeps both schedules.
    two_stage = _template("dgx-monarch-test-ltx25-two-stage")
    convert.set_probe_steps(two_stage, 1)
    assert sorted(node["inputs"]["sigmas"] for node in two_stage.values()
                  if node["class_type"] == "ManualSigmas") == ["0.85, 0.0", "1.0, 0.0"]


def test_every_shipped_template_carries_a_step_count_the_rewrite_can_set():
    # The table is what decides a probe leg, so a template whose scheduler it
    # does not name silently loses one. This is the list that says none do.
    templates = sorted(list(TEMPLATE_DIR.glob("*.json"))
                       + list((REPO / "tests" / "fixtures" / "workflows" / "generated").glob("*.json")))
    with _generator_widget_names() as names:
        missing = {path.stem: convert.probe_support(
            convert.convert(json.loads(path.read_text()), names))[1] for path in templates}
    assert templates and not {stem: why for stem, why in missing.items() if why}


def test_a_step_count_the_table_does_not_know_keeps_the_graph_off_the_probe():
    # Half a rewritten schedule is a graph neither leg renders, so one node the
    # table misses drops the probe for the whole graph, naming the type.
    known = {"1": {"class_type": "DGXMonarchKSampler", "inputs": {"steps": 20}}}
    assert convert.probe_support(known) == (True, "")
    unknown = dict(known, **{"2": {"class_type": "FutureScheduler",
                                   "inputs": {"steps": 20, "width": 1024}}})
    ok, why = convert.probe_support(unknown)
    assert not ok and "FutureScheduler" in why
    # A type the table does name, whose widget is a link: the rewrite knows
    # where the count lives and still cannot set it. A check that misses it
    # hands stage two a full schedule under a probe leg.
    linked_stage = {"3": {"class_type": "DGXMonarchKSamplerAdvanced",
                          "inputs": {"steps": ["9", 0], "start_at_step": 0,
                                     "end_at_step": 20}}}
    ok, why = convert.probe_support(dict(known, **linked_stage))
    assert not ok and why == ("DGXMonarchKSamplerAdvanced reads its step count off a link, "
                              "which the probe rewrite cannot set")
    # A graph carrying both is told both, so the fix names every node to chase.
    ok, why = convert.probe_support(dict(unknown, **linked_stage))
    assert not ok and "FutureScheduler" in why and "DGXMonarchKSamplerAdvanced" in why
    # A lone linked carrier leaves the graph with no settable count at all.
    linked = {"1": {"class_type": "ManualSigmas", "inputs": {"sigmas": ["9", 0]}}}
    assert convert.probe_support(linked) == (
        False, "ManualSigmas reads its step count off a link, "
        "which the probe rewrite cannot set")


def test_a_cell_with_no_probe_leg_is_scored_at_the_full_render_bar(tmp_path):
    # ideogram4 bf16 uly2 read 0.198 as a full render and 0.000 at one step
    # (2026-09-03), so the one-step floor is the wrong bar for a full render.
    config = _rig(tmp_path)
    graph = _graph("krea2_raw_bf16.safetensors")
    graph["4"]["inputs"].pop("steps")
    graph["6"] = {"class_type": "FutureScheduler", "inputs": {"steps": 20}}
    (config["out_dir"] / "graphs" / "future.json").write_text(json.dumps(graph))
    cells, pruned, _unmatched = matrix.build_cells(config)
    cell = _pick(cells, "future", "uly2", attention="TORCH_FLASH")
    assert cell["probe"] is False and "FutureScheduler" in cell["probe_reason"]
    assert cell["reference"]["bar"] == matrix.FULL_RENDER_BAR
    assert cell["reference"]["probe_steps"] == 0
    assert any("future: no one-step probe leg" in line and "FutureScheduler" in line
               for line in pruned)
    # The runner plans no probe leg for it, and the local reference plans none
    # either, so nothing renders a leg the score cannot use.
    assert "probe" not in [leg for leg, *_rest in run.leg_plan(cell)]
    # A graph the table does know keeps its probe leg and the step-nrms bar.
    scored = _pick(cells, "krea2", "uly2", artifact_quant="bf16", attention="TORCH_FLASH")
    assert scored["probe"] and scored["probe_reason"] == ""
    assert scored["reference"]["bar"] == "step-nrms"
    assert "probe" in [leg for leg, *_rest in run.leg_plan(scored)]


def test_a_residency_or_inert_lever_is_held_to_the_pixel(tmp_path):
    # Neither kind changes an arithmetic result: one moves where the weights
    # live, the other moves when the work runs. A difference between the two
    # renders is therefore the finding, which is why the bar is the pixel.
    cells, _pruned, _unmatched = matrix.build_cells(_levers(tmp_path))
    auto = _lever(cells, "krea2", {}, artifact_quant="bf16", attention="TORCH_FLASH")
    for levers in ({"slab_weights": "on"}, {"pipeline_depth": 2},
                   {"comfy_managed": "on", "pipeline_depth": 2}):
        cell = _lever(cells, "krea2", levers, artifact_quant="bf16", attention="TORCH_FLASH")
        assert cell["reference"] == {"kind": "cell", "cell": auto["id"],
                                     "bar": "bit-identical"}, levers
        # The whole warm render is the comparison, so no probe leg is planned:
        # one step would prove less of it than all of it does.
        assert "probe" not in [leg for leg, *_rest in run.leg_plan(cell)], levers


def test_a_numeric_lever_is_scored_probe_against_probe(tmp_path):
    # Scored at the full-render bar, 1026 cells of the rig matrix could never
    # PASS (2026-09-03), and a sweep that cannot pass reports no verdict.
    cells, _pruned, _unmatched = matrix.build_cells(_levers(tmp_path))
    by_id = {cell["id"]: cell for cell in cells}
    auto = _lever(cells, "krea2", {}, artifact_quant="bf16", attention="TORCH_FLASH")
    # A lever set holding one numeric lever is numeric, whatever rides with it.
    for levers in ({"compile_dit": True}, {"slab_weights": "on", "compile_dit": True}):
        cell = _lever(cells, "krea2", levers, artifact_quant="bf16", attention="TORCH_FLASH")
        assert cell["reference"] == {"kind": "cell", "cell": auto["id"],
                                     "bar": "step-nrms", "probe_steps": 1}, levers
        assert "probe" in [leg for leg, *_rest in run.leg_plan(cell)], levers
    # Both sides of the comparison render one, or the candidate has nothing to
    # be measured against.
    assert "probe" in [leg for leg, *_rest in run.leg_plan(by_id[auto["id"]])]


def test_a_numeric_lever_the_probe_cannot_cut_keeps_the_full_render_bar(tmp_path):
    # The rewrite has to reach the step count of both legs. Where it cannot,
    # the comparison is two full renders and the one-step floor never scores it.
    config = _levers(tmp_path)
    graph = _graph("krea2_raw_bf16.safetensors")
    graph["4"]["inputs"].pop("steps")
    graph["6"] = {"class_type": "FutureScheduler", "inputs": {"steps": 20}}
    (config["out_dir"] / "graphs" / "future.json").write_text(json.dumps(graph))
    cells, _pruned, _unmatched = matrix.build_cells(config)
    auto = _lever(cells, "future", {}, attention="TORCH_FLASH")
    cell = _lever(cells, "future", {"compile_dit": True}, attention="TORCH_FLASH")
    assert cell["reference"] == {"kind": "cell", "cell": auto["id"],
                                 "bar": matrix.FULL_RENDER_BAR, "probe_steps": 0}
    assert "probe" not in [leg for leg, *_rest in run.leg_plan(cell)]
    # The runner's finding names the node type off this, so the next template
    # landing a scheduler of its own says which one the table has to learn.
    assert "FutureScheduler" in cell["probe_reason"]
    # An inert lever on the same template is still held to the pixel: it never
    # needed a probe leg, so a graph the rewrite cannot cut costs it nothing.
    inert = _lever(cells, "future", {"pipeline_depth": 2}, attention="TORCH_FLASH")
    assert inert["reference"]["bar"] == "bit-identical"


def test_a_sol_kernel_records_its_deviation_instead_of_checking_it(tmp_path):
    # The kernel is a known-wrong approximation admitted under a class K
    # waiver, so the deviation is what the cell was derived to measure. A CHECK
    # would file the result of the cell as a doubt about it.
    cells, _pruned, _unmatched = matrix.build_cells(_levers(tmp_path))
    by_id = {cell["id"]: cell for cell in cells}
    waived = _pick(cells, "h3", "uly2", attention="SOL_ATTN_TAU0.7")
    assert (waived["label"], waived["waiver"]) == ("refuse:K", True)
    assert waived["reference"]["bar"] == matrix.WAIVED_BAR
    # Measured at one step like any other comparison, against the base kernel's
    # own render: the kernel is the difference the number reports.
    assert waived["reference"]["probe_steps"] == 1
    assert by_id[waived["reference"]["cell"]]["attention"] == "TORCH_FLASH"
    # A numeric lever on a sol cell reads the same way: the waived bar outranks
    # the step-nrms floor, because the floor cannot answer for this kernel.
    lever = _lever(cells, "h3", {"compile_dit": True}, attention="SOL_ATTN_TAU0.7")
    assert lever["reference"]["bar"] == matrix.WAIVED_BAR
    # Where every lever is inert or residency, both sides run the same kernel,
    # so the stricter bar stands and the waiver changes nothing.
    inert = _lever(cells, "h3", {"pipeline_depth": 2}, attention="SOL_ATTN_TAU0.7")
    assert inert["reference"]["bar"] == "bit-identical"
    # known_wrong_kernel reads the waiver guards as well as the kernel name, so
    # a kernel that earns the same waiver under a later name lands on the same bar.
    assert matrix.known_wrong_kernel({"attention": "TORCH_FLASH",
                                      "waiver_guards": ["ring_pad", "sol_attn"]})
    assert not matrix.known_wrong_kernel({"attention": "TORCH_FLASH",
                                          "waiver_guards": ["ring_pad"]})


def test_the_probe_leg_of_a_lever_cell_carries_the_lever(tmp_path):
    # A probe leg that dropped the lever would render the auto cell twice and
    # read 0.000 for every numeric lever in the wave.
    config = _levers(tmp_path)
    cells, _pruned, _unmatched = matrix.build_cells(config)
    cell = _lever(cells, "krea2", {"compile_dit": True},
                  artifact_quant="bf16", attention="TORCH_FLASH")
    graph = json.loads((config["out_dir"] / "graphs" / "krea2.json").read_text())
    steps = {leg: count for leg, _preset, _gate, _shift, count in run.leg_plan(cell)}
    assert steps["probe"] == 1
    probe = run.patch_graph(graph, cell, cell["preset"], cell["auto_gate"], 0,
                            "sweep_x_probe", steps["probe"])
    assert probe["1"]["inputs"]["compile_dit"] is True
    assert probe["1"]["inputs"]["topology"] == cell["preset"]
    assert probe["4"]["inputs"]["steps"] == 1
    # The warm leg beside it carries the same lever over the full schedule.
    warm = run.patch_graph(graph, cell, cell["preset"], cell["auto_gate"], 1, "sweep_x_warm")
    assert warm["1"]["inputs"]["compile_dit"] is True and warm["4"]["inputs"]["steps"] == 20


def test_a_moved_reference_reruns_a_settled_record():
    # A moved label re-runs a record because the verdict was scored against it.
    # A moved reference re-runs it too: it names the legs the cell renders and
    # the bar they are scored at.
    pin = dict.fromkeys(run.RESUME_KEYS, "pinned")
    reference = {"kind": "cell", "cell": "abc123",
                 "bar": matrix.FULL_RENDER_BAR, "probe_steps": 0}
    done = {"pin": dict(pin), "graph_digest": "g1", "verdict": "CHECK", "observed": "render",
            "cell": {"label": "render", "reference": dict(reference)}}
    assert run.resumable(done, pin, "g1", "render", reference)
    for moved in ({"probe_steps": 1, "bar": "step-nrms"}, {"cell": "def456"},
                  {"kind": "none"}, {"bar": "bit-identical"}):
        assert not run.resumable(done, pin, "g1", "render", dict(reference, **moved)), moved
    # A key outside the list decides nothing: a resident-leg preset is cut from
    # the cell's own preset, which is part of the cell id.
    assert run.resumable(done, pin, "g1", "render", dict(reference, preset="uly2"))
    # Every record written before the reference was read re-runs once.
    assert not run.resumable({**done, "cell": {"label": "render"}}, pin, "g1",
                             "render", reference)


def test_a_full_render_check_is_listed_apart_from_a_probe_check():
    # A number measured against no floor reads nothing like one over a
    # calibrated floor, and on this matrix it outnumbers them.
    from benchmark.sweep import report

    measured = (f"{matrix.FULL_RENDER_BAR} 0.198 against sweep_b_warm_00001_.png: no probe "
                "leg is available on this cell, so no floor applies; this reference spec "
                "plans no probe leg")
    records = [
        {"cell": {"id": "a", "session": "cluster", "label": "render",
                  "template": "ltx", "preset": "uly2", "attention": "TORCH_FLASH"},
         "verdict": "CHECK", "full_render": measured, "findings": [measured]},
        {"cell": {"id": "c", "session": "cluster", "label": "render",
                  "template": "krea2", "preset": "uly2", "attention": "TORCH_FLASH"},
         "verdict": "CHECK", "findings": ["step-nrms 0.14 over the 0.1 floor against "
                                          "sweep_d_probe_00001_.png"]},
        # A cell holding both: the fault belongs in the findings list and the
        # measurement below it, so the split reads the line and not the record.
        {"cell": {"id": "e", "session": "cluster", "label": "render",
                  "template": "wan22", "preset": "uly2", "attention": "TORCH_FLASH"},
         "verdict": "FINDING", "full_render": measured,
         "findings": ["warm leg ran in 2.0 s against a 40.0 s cold leg; "
                      "the execution cache probably served it", measured]},
    ]
    payload = report.build(records, "cluster")
    assert len(payload["findings"]) == 2
    assert "step-nrms 0.14" in payload["findings"][0]
    assert "execution cache" in payload["findings"][1]
    assert [line.split("`")[1] for line in payload["full_render"]] == ["a", "e"]
    text = report.render(payload)
    head, tail = text.split("## full-render measurements")
    assert head.index("## findings") < len(head)
    assert "step-nrms 0.14" in head and "execution cache" in head
    assert "0.198" in tail and "0.198" not in head


def test_a_waived_kernel_number_is_listed_under_its_own_heading():
    # It never reaches the findings list, because the waiver admits it. Left
    # to the table's nrms column alone, nothing would say what it measured.
    from benchmark.sweep import report

    measured = (f"{matrix.WAIVED_BAR} 0.42 against sweep_ref_probe_00001_.png: "
                "SOL_ATTN_TAU0.7 is an approximation the class K waiver admits, so "
                "the deviation is the measurement and no floor applies")
    records = [
        {"cell": {"id": "s", "session": "cluster", "label": "refuse:K", "template": "h3",
                  "preset": "uly2", "attention": "SOL_ATTN_TAU0.7"},
         "verdict": "PASS", "waived_render": measured, "findings": [], "notes": [measured]},
        {"cell": {"id": "c", "session": "cluster", "label": "render", "template": "krea2",
                  "preset": "uly2", "attention": "TORCH_FLASH"},
         "verdict": "CHECK", "findings": ["step-nrms 0.14 over the 0.1 floor"]},
    ]
    payload = report.build(records, "cluster")
    assert [line.split("`")[1] for line in payload["waived"]] == ["s"]
    assert len(payload["findings"]) == 1 and payload["full_render"] == []
    head, tail = report.render(payload).split("## waived-render measurements")
    assert "0.42" in tail and "0.42" not in head
    assert "step-nrms 0.14" in head


def test_a_waived_leg_number_is_listed_under_the_same_heading():
    # A cell the matrix could only label refuse:K carries the same kind of
    # number: the card was granted and the leg rendered under it. The report
    # collects it off the same key, so the heading learns no second rule.
    from benchmark.sweep import report

    measured = (f"{matrix.WAIVED_BAR} 0.120494 against sweep_ref_probe_00001_.png: the "
                "runner granted the class K card for shard_quant_scale:chroma and this leg "
                "rendered under it, so the deviation is the measurement and no floor applies")
    records = [
        {"cell": {"id": "w", "session": "cluster", "label": "refuse:K", "template": "chroma",
                  "preset": "uly2", "attention": "TORCH_FLASH"},
         "verdict": "PASS", "waived_render": measured, "findings": [], "notes": [measured]},
        {"cell": {"id": "c", "session": "cluster", "label": "render", "template": "krea2",
                  "preset": "uly2", "attention": "TORCH_FLASH"},
         "verdict": "CHECK", "findings": ["step-nrms 0.14 over the 0.1 floor"]},
    ]
    payload = report.build(records, "cluster")
    assert [line.split("`")[1] for line in payload["waived"]] == ["w"]
    assert len(payload["findings"]) == 1 and payload["full_render"] == []
    head, tail = report.render(payload).split("## waived-render measurements")
    assert "0.120494" in tail and "0.120494" not in head


def _ideogram_rig(tmp_path: Path) -> dict:
    """A rig carrying ideogram4, the one family whose head dimension cuDNN and sage cannot carry.

    It places both loaders: ideogram4 guidance runs a separate unconditional
    model, and a cfg cell without the second checkpoint refuses for that reason
    instead of the one under test.
    """
    config = _rig(tmp_path)
    models = Path(config["comfy_dir"]) / "models" / "diffusion_models"
    for unet in ("ig4_bf16.safetensors", "ig4_uncond_bf16.safetensors"):
        _checkpoint(models, unet, "ideogram4")
    graph = _graph("ig4_bf16.safetensors")
    graph["6"] = {"class_type": "DGXMonarchUncondUNETLoader",
                  "inputs": {"unet_name": "ig4_uncond_bf16.safetensors", "mesh": ["1", 0]}}
    (Path(config["out_dir"]) / "graphs" / "ig4.json").write_text(json.dumps(graph))
    config["image_attention"] = ["TORCH_FLASH", "TORCH_CUDNN", "SAGE_AUTO"]
    return config


def _ig4(cells: list[dict], preset: str) -> list[dict]:
    return [cell for cell in cells if cell["template"] == "ig4"
            and cell["preset"] == preset and cell["session"] == "cluster"]


def test_a_kernel_that_cannot_carry_the_head_dimension_gives_way_to_flash(tmp_path):
    """Seven Ideogram4 runs crashed inside the ring on TORCH_CUDNN and SAGE_AUTO
    (docs/VALIDATION.md, 2026-09-04). Every sharded cell
    resolves to the kernel that carries head dimension 256 and collapses onto
    it. cfg2 installs no sharded attention and keeps the kernel it names, and
    both tested cfg2 runs completed with their selected kernels."""
    cells, pruned, _unmatched = matrix.build_cells(_ideogram_rig(tmp_path))
    for preset in ("auto", "uly2", "ring2", "uly2+fsdp"):
        sharded = _ig4(cells, preset)
        assert len(sharded) == 1, [cell["attention"] for cell in sharded]
        assert sharded[0]["resolved_attention"] == "TORCH_FLASH", preset
    for preset in ("auto", "uly2", "ring2"):
        assert _ig4(cells, preset)[0]["label"] == "render", preset
    # uly2+fsdp answers earlier and untagged: the dual-model clean-reload gate
    # sits above every sample-time question and is a separate issue.
    assert _ig4(cells, "uly2+fsdp")[0]["label"] == "refuse:untyped"
    by_kernel = {cell["attention"]: cell for cell in _ig4(cells, "cfg2")}
    assert set(by_kernel) == {"TORCH_FLASH", "TORCH_CUDNN", "SAGE_AUTO"}
    for kernel, cell in by_kernel.items():
        assert (cell["resolved_attention"], cell["label"]) == (kernel, "render")
    assert any("head dimension" in line for line in pruned), pruned
    # A family that declares no head dimension keeps the kernel it names.
    krea2 = _pick(cells, "krea2", "uly2", artifact_quant="bf16", attention="TORCH_CUDNN")
    assert (krea2["resolved_attention"], krea2["label"]) == ("TORCH_CUDNN", "render")


def test_a_head_dimension_no_kernel_carries_still_refuses(tmp_path, monkeypatch):
    """The substitution is not a blanket pass: with nothing to swap to, the
    published-limit check answers class P and the cell says so."""
    monkeypatch.setattr(matrix.ADAPTER_BY_FAMILY["ideogram4"],
                        "attention_head_dim", 512)
    cells, _pruned, _unmatched = matrix.build_cells(_ideogram_rig(tmp_path))
    for kernel in ("TORCH_CUDNN", "SAGE_AUTO", "TORCH_FLASH"):
        cell = _pick(cells, "ig4", "uly2", attention=kernel)
        assert (cell["label"], cell["basis"]) == ("refuse:P", "guard"), kernel
    assert _pick(cells, "ig4", "cfg2", attention="TORCH_CUDNN")["label"] == "render"
