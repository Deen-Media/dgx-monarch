"""Every bundled template opens on a reviewed App Mode layout in Studio and ComfyUI.

tools/gen_templates.py writes each template's layout into extra.linearData as
[node id, widget name] pairs and the ids of its save nodes. This module checks
each layout against its template's own graph, never against the generator's
table. It works out from the nodes and wires which inputs the approved rule asks
for, and the layout must list exactly those, in the approved order, apart from
the reviewed exceptions below, each with its reason. Studio inspects every
template against the recorded driver schema, so each entry must also name an
editable input that exists.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from dgx_monarch.studio.workflows import binding_id, inspect_workflow

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "example_workflows").glob("*.json"))
OBJECT_INFO = json.loads((ROOT / "tests/fixtures/studio/object_info_bundled.json").read_text())

# The approved order: the positive prompt, the negative prompt, the seed, the
# image size, steps, cfg, sampler and scheduler, the denoise or strength of an
# edit or upscale, the input media, the LoRA strength and the model file.
ROLES = ("prompt", "negative", "seed", "size", "steps", "cfg", "sampler", "strength",
         "media", "lora", "model")
# Text inputs whose side, positive or negative, the wires decide.
TEXT = {"text", "clip_l", "qwen25_7b", "value"}
# Flux Guidance's guidance is the scale a guidance distilled model reads in
# place of cfg, so it shares the cfg role.
BY_NAME = {"prompt": "prompt", "prompts": "prompt", "negative_prompt": "negative",
           "seed": "seed", "noise_seed": "seed", "width": "size", "height": "size",
           "resolution": "size", "steps": "steps", "cfg": "cfg", "guidance": "cfg",
           "sampler_name": "sampler", "scheduler": "sampler", "denoise": "strength",
           "degrade_sigma": "strength", "strength_model": "lora", "unet_name": "model"}
MEDIA = {("LoadImage", "image"), ("LoadImageMask", "image"), ("LoadVideo", "file"),
         ("LoadAudio", "audio")}
# Their inputs stay under All inputs.
LEFT_OUT_NODES = {"DGXMonarchInit", "DGXMonarchSkipLayerGuidance", "VAELoader", "CLIPLoader",
                  "DualCLIPLoader", "CLIPVisionLoader"}
# A graph that reads or writes one of these makes video or sound, not a still image.
VIDEO_OR_SOUND = {"LoadVideo", "LoadAudio", "SaveVideo", "SaveAudioAdvanced"}
# A sampler that takes an image from one of these edits or refines that image.
IMAGE_SOURCES = {"LoadImage", "LoadImageMask", "VAEEncode"}
# The length of a node that also sets a width and a height: the clip's frame count.
FRAME_COUNT = "frame count"

# The reviewed exceptions. The rule asks for these inputs, and the layouts
# leave them out on purpose: by kind in every template, or one at a time by
# template, node id and input.
OMITTED_KINDS = {
    FRAME_COUNT: ("Frame counts stay under All inputs: several video graphs repeat the count "
                  "on another node that must change with it, such as the LTX audio latent, "
                  "the MiniMax conditioning node and the WanDancer audio encode."),
}
NOISE_FREE = ("The low-noise stage runs with add_noise disabled and a sampler that draws no "
              "noise, so its seed changes nothing.")
EMPTY_LATENT = ("The edit samples from an empty latent and takes the reference through its "
                "conditioning, so a denoise below 1 damages the render instead of weakening "
                "the edit.")
EMPTY_CANVAS = ("PiD samples from an empty canvas, so its denoise has nothing to keep; the "
                "layout lists degrade_sigma instead.")
OMITTED = {
    ("dgx-monarch-wan22-t2v", 12, "noise_seed"): NOISE_FREE,
    ("dgx-monarch-wan22-i2v", 14, "noise_seed"): NOISE_FREE,
    ("dgx-monarch-wan-bernini", 13, "noise_seed"): NOISE_FREE,
    ("dgx-monarch-flux3-klein-edit", 14, "denoise"): EMPTY_LATENT,
    ("dgx-monarch-flux3-klein-base-edit", 14, "denoise"): EMPTY_LATENT,
    ("dgx-monarch-mage-flow-edit", 8, "denoise"): EMPTY_LATENT,
    ("dgx-monarch-mage-flow-edit-turbo", 8, "denoise"): EMPTY_LATENT,
    ("dgx-monarch-qwen-image21-edit-rgba", 27, "denoise"): EMPTY_LATENT,
    ("dgx-monarch-qwen-image21-edit-mask-rgba", 10, "denoise"): EMPTY_LATENT,
    ("dgx-monarch-pid-4k", 13, "denoise"): EMPTY_CANVAS,
}
# The layouts list these inputs although the rule does not ask for them.
ADDED = {
    ("dgx-monarch-pid-4k", 11, "degrade_sigma"): (
        "PiD's degrade_sigma takes the place of the denoise its empty canvas cannot use: it "
        "sets how degraded PiD takes the source to be, and 0 treats the source as clean."),
}


def side(workflow: dict, node_id: int) -> str | None:
    """Whether a text node feeds a positive or a negative socket.

    Walk the CONDITIONING and STRING wires outward from the node, one hop at a
    time. The nearest socket whose name says positive or negative decides; a
    tie, or no such socket, gives None. Conditioning Zero Out keeps no text, so
    the walk ends there: a prompt that reaches the sampler only through it, as
    on the Fleet graph, is no prompt.
    """
    nodes = {node["id"]: node for node in workflow["nodes"]}
    frontier, seen = [node_id], {node_id}
    while frontier:
        sides, following = set(), []
        for _link, origin, _slot, target, target_slot, kind in workflow["links"]:
            if origin not in frontier or kind not in {"CONDITIONING", "STRING"}:
                continue
            if nodes[target]["type"] == "ConditioningZeroOut":
                continue
            name = nodes[target]["inputs"][target_slot]["name"]
            if "negative" in name:
                sides.add("negative")
            elif "positive" in name:
                sides.add("prompt")
            elif target not in seen:
                seen.add(target)
                following.append(target)
        if sides:
            return sides.pop() if len(sides) == 1 else None
        frontier = following
    return None


def upstream(workflow: dict, node_id: int) -> set[int]:
    """Every node whose output reaches this node through the wires."""
    found: set[int] = set()
    pending = [node_id]
    while pending:
        target = pending.pop()
        for _link, origin, _slot, end, _target_slot, _kind in workflow["links"]:
            if end == target and origin not in found:
                found.add(origin)
                pending.append(origin)
    return found


def role(workflow: dict, node: dict, name: str, inputs: set[str]) -> str | None:
    """The approved role of one input, FRAME_COUNT, or None for an input no layout lists.

    `inputs` holds the names of every typed input of the node, wired or not.
    """
    if node["type"] in LEFT_OUT_NODES:
        return None
    if name in TEXT:
        return side(workflow, node["id"])
    if (node["type"], name) in MEDIA:
        return "media"
    if name == "length" and {"width", "height"} <= inputs:
        return FRAME_COUNT
    return BY_NAME.get(name)


def asked_for(workflow: dict, bindings: list[dict]) -> dict[tuple[int, str], str]:
    """The inputs the approved rule asks a layout to list, each with its role.

    It reads the graph alone. A wired input holds no value to edit, so the rule
    never asks for one. The rule asks for a denoise only where it is the
    strength of an edit or an upscale: on a sampler of a still image graph that
    takes an image the graph loads or encodes. degrade_sigma is no denoise, so
    the rule never asks for it.
    """
    nodes = {node["id"]: node for node in workflow["nodes"]}
    rows = [row for row in bindings if row["node_id"] is not None]
    names: dict[int, set[str]] = {}
    for row in rows:
        names.setdefault(int(row["node_id"]), set()).add(row["input_name"])
    still = (not any(node["type"] in VIDEO_OR_SOUND for node in nodes.values())
             and not any({"length", "width", "height"} <= found for found in names.values()))
    asked = {}
    for row in rows:
        node_id, name = int(row["node_id"]), row["input_name"]
        found = role(workflow, nodes[node_id], name, names[node_id])
        if not row["resolved"] or found is None:
            continue
        if found == "strength" and not (
                name == "denoise" and still
                and {nodes[source]["type"] for source in upstream(workflow, node_id)} & IMAGE_SOURCES):
            continue
        asked[(node_id, name)] = found
    return asked


def layout_problems(stem: str, workflow: dict) -> list[str]:
    """Every way a template's layout departs from the rule; empty when it holds."""
    layout = workflow.get("extra", {}).get("linearData") or {}
    entries = [tuple(entry) for entry in layout.get("inputs", [])]
    outputs = list(layout.get("outputs", []))
    if not entries or not outputs:
        return [f"{stem}: the layout lists no input or no output"]
    nodes = {node["id"]: node for node in workflow["nodes"]}
    inspection = inspect_workflow(workflow, OBJECT_INFO)
    problems = [f"{stem}: {error}" for error in inspection["errors"]]
    rows = [row for row in inspection["bindings"] if row["node_id"] is not None]
    by_key = {(int(row["node_id"]), row["input_name"]): row for row in rows}
    names: dict[int, set[str]] = {}
    for node_id, name in by_key:
        names.setdefault(node_id, set()).add(name)

    # Each entry names an editable input that exists, and Studio shows exactly
    # those entries as prominent, in layout order.
    if len(set(entries)) != len(entries):
        problems.append(f"{stem}: the layout lists an input twice")
    for node_id, name in entries:
        if not by_key.get((node_id, name), {}).get("resolved"):
            problems.append(f"{stem}: {node_id}.{name} is no editable input of the graph")
    expected = [binding_id(workflow["id"], str(node_id), name) for node_id, name in entries]
    prominent = [row["id"] for row in sorted((row for row in inspection["bindings"] if row["prominent"]),
                                             key=lambda row: row["order"])]
    if prominent != expected:
        problems.append(f"{stem}: Studio shows {prominent} as prominent, not the layout's {expected}")

    # Only approved inputs, in the approved order, starting with prompt text.
    roles = [role(workflow, nodes[node_id], name, names.get(node_id, set())) if node_id in nodes else None
             for node_id, name in entries]
    for (node_id, name), found in zip(entries, roles, strict=True):
        if found not in ROLES:
            problems.append(f"{stem}: {node_id}.{name} has no approved role")
    ranks = [ROLES.index(found) for found in roles if found in ROLES]
    if ranks != sorted(ranks):
        problems.append(f"{stem}: the roles run out of order: {roles}")
    first = by_key.get(entries[0], {})
    if roles[0] != "prompt" or (first.get("schema") or {}).get("type") != "STRING":
        problems.append(f"{stem}: the layout does not start with prompt text")

    # Exactly the inputs the rule asks for, apart from the reviewed exceptions.
    asked = asked_for(workflow, rows)
    listed = set(entries)
    omitted = {(node_id, name) for (template, node_id, name) in OMITTED if template == stem}
    omitted |= {key for key, found in asked.items() if found in OMITTED_KINDS}
    added = {(node_id, name) for (template, node_id, name) in ADDED if template == stem}
    for node_id, name in sorted(set(asked) - listed - omitted):
        problems.append(f"{stem}: the rule asks for {nodes[node_id]['type']} input "
                        f"{node_id}.{name}, and the layout leaves it out")
    for node_id, name in sorted(listed - set(asked) - added):
        problems.append(f"{stem}: the layout lists {node_id}.{name}, which the rule does not ask for")
    for node_id, name in sorted(listed & omitted):
        problems.append(f"{stem}: the layout lists {node_id}.{name}, which an exception leaves out")
    for node_id, name in sorted(omitted - set(asked)):
        problems.append(f"{stem}: the exception for {node_id}.{name} names an input the rule does not ask for")
    for node_id, name in sorted(added - listed):
        problems.append(f"{stem}: the layout leaves out {node_id}.{name}, which an exception adds")
    for node_id, name in sorted(added & set(asked)):
        problems.append(f"{stem}: the exception adding {node_id}.{name} names an input the rule asks for")

    # The app view shows every file the template saves, and nothing else.
    saves = sorted(node_id for node_id, node in nodes.items() if node["type"].startswith("Save"))
    if sorted(outputs) != saves:
        problems.append(f"{stem}: the outputs are {sorted(outputs)}, not the save nodes {saves}")
    if not all(row["resolved"] for row in inspection["outputs"]):
        problems.append(f"{stem}: an output does not resolve")
    return problems


def _template(stem: str) -> dict:
    return json.loads((ROOT / "example_workflows" / f"{stem}.json").read_text())


def test_the_template_folder_is_not_empty():
    # The parametrized checks pass on an empty folder, so this one fails on it.
    assert TEMPLATES


@pytest.mark.parametrize("path", TEMPLATES, ids=[path.stem for path in TEMPLATES])
def test_template_opens_on_its_app_mode_layout(path: Path):
    assert layout_problems(path.stem, json.loads(path.read_text())) == []


@pytest.mark.parametrize("path", TEMPLATES, ids=[path.stem for path in TEMPLATES])
def test_dropping_any_entry_fails_the_check(path: Path):
    """Each entry is needed: drop any one, and the check names it.

    This covers a layout that drops its seed, its negative prompt or its input
    image but keeps its positive prompt.
    """
    workflow = json.loads(path.read_text())
    for index, (node_id, name) in enumerate(workflow["extra"]["linearData"]["inputs"]):
        mutated = copy.deepcopy(workflow)
        del mutated["extra"]["linearData"]["inputs"][index]
        problems = layout_problems(path.stem, mutated)
        assert any(f" {node_id}.{name}" in problem for problem in problems), (node_id, name, problems)


@pytest.mark.parametrize(("stem", "entry"), [
    ("dgx-monarch-chroma-t2i", (7, "batch_size")),
    ("dgx-monarch-chroma-t2i", (9, "vae_name")),
    ("dgx-monarch-chroma-t2i", (8, "denoise")),
    ("dgx-monarch-wan-t2v", (8, "length")),
    ("dgx-monarch-lora-low-rss", (9, "lora_name")),
    ("dgx-monarch-wan22-t2v", (12, "noise_seed")),
], ids=["batch-size", "vae-file", "text-to-image-denoise", "frame-count", "lora-file",
        "noise-free-seed"])
def test_listing_an_input_the_rule_leaves_out_fails_the_check(stem: str, entry: tuple[int, str]):
    workflow = _template(stem)
    inputs = workflow["extra"]["linearData"]["inputs"]
    inputs.insert(len(inputs) - 1, list(entry))
    problems = layout_problems(stem, workflow)
    assert any(f" {entry[0]}.{entry[1]}" in problem for problem in problems), problems


def test_every_exception_names_a_template_and_its_reason_holds_in_the_graph():
    # layout_problems checks that each exception names an input of its
    # template; this test checks the template exists and the reason is true.
    stems = {path.stem for path in TEMPLATES}
    for (stem, node_id, _name), reason in {**OMITTED, **ADDED}.items():
        assert stem in stems, stem
        workflow = _template(stem)
        node = next(node for node in workflow["nodes"] if node["id"] == node_id)
        if reason == NOISE_FREE:
            assert node["dgxm_widgets"]["add_noise"] == "disable", (stem, node_id)
        if reason in (EMPTY_LATENT, EMPTY_CANVAS):
            # The sampler's latent comes from no VAE Encode, so it holds no image.
            slot = next(index for index, socket in enumerate(node["inputs"])
                        if socket["name"] == "latent_image")
            source = next(origin for _link, origin, _slot, target, target_slot, _kind in workflow["links"]
                          if (target, target_slot) == (node_id, slot))
            latent = next(item for item in workflow["nodes"] if item["id"] == source)
            assert latent["type"] != "VAEEncode", (stem, latent["type"])
            if latent["type"] == "LatentMultiply":
                assert latent["widgets_values"] == [0.0], (stem, latent["widgets_values"])


def test_side_follows_conditioning_wires_to_the_sampler_socket():
    # The Flux 2 prompt reaches the sampler through Flux Guidance, and its
    # Conditioning Zero Out copy is the negative; the walk ends at the zero
    # out, so the text is the positive prompt.
    workflow = _template("dgx-monarch-flux2-t2i")
    by_type = {node["type"]: node["id"] for node in workflow["nodes"]}
    assert side(workflow, by_type["CLIPTextEncode"]) == "prompt"
    wan = _template("dgx-monarch-wan-t2v")
    texts = [node["id"] for node in wan["nodes"] if node["type"] == "CLIPTextEncode"]
    assert [side(wan, node_id) for node_id in texts] == ["prompt", "negative"]
    # The Fleet sampler takes its prompts itself; its text encode only feeds
    # the zeroed negative, so it is no prompt.
    fleet = _template("dgx-monarch-fleet")
    encode = next(node["id"] for node in fleet["nodes"] if node["type"] == "CLIPTextEncode")
    assert side(fleet, encode) is None


@pytest.fixture(scope="module")
def generator():
    original = list(sys.path)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import gen_templates
    finally:
        sys.path[:] = original
    return gen_templates


def test_every_shipped_template_has_a_layout_and_no_testing_graph_does(generator):
    assert set(generator.APP_MODE) == set(generator.TEMPLATES)
    assert all("app_mode" in spec for spec in generator.TEMPLATES.values())
    assert not any("app_mode" in spec for spec in generator.TESTING_TEMPLATES.values())


@pytest.mark.parametrize(("inputs", "outputs", "error", "message"), [
    (["nope.text"], ["save"], KeyError, "names no node"),
    (["pos.prompt"], ["save"], KeyError, "has no widget 'prompt'"),
    (["ks.control_after_generate"], ["save"], KeyError, "has no widget"),
    (["init.topology"], ["save"], ValueError, "stays under All inputs"),
    (["clip.clip_name1"], ["save"], ValueError, "stays under All inputs"),
    (["pos.text", "pos.text"], ["save"], ValueError, "listed twice"),
    (["pos.text"], ["dec"], ValueError, "not a save node"),
    ([], ["save"], ValueError, "at least one input"),
], ids=["missing-node", "missing-widget", "control-widget", "init", "text-encoder",
        "repeat", "not-a-save", "empty"])
def test_build_refuses_a_layout_the_graph_cannot_show(generator, inputs, outputs, error, message):
    spec = {**generator.TEMPLATES["dgx-monarch-quickstart"],
            "app_mode": {"inputs": inputs, "outputs": outputs}}
    with pytest.raises(error, match=message):
        generator.build(spec)


def test_build_refuses_a_wired_widget_and_an_unknown_role(generator):
    # The MOSS-Image layout graph wires its JSON into CLIP Text Encode's text.
    spec = {**generator.TEMPLATES["dgx-monarch-moss-image-layout-t2i"],
            "app_mode": {"inputs": ["pos.text"], "outputs": ["save"]}}
    with pytest.raises(ValueError, match="is wired"):
        generator.build(spec)
    with pytest.raises(KeyError, match="colour"):
        generator._layout({"colour": ("pos.text",)})
