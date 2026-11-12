"""The MiniMax-H3-Image templates keep upstream's recipe, and their Notes match the code.

ComfyUI's own templates for this family (Comfy-Org/workflow_templates 0.11.105,
image_minimax_h3_image_t2i and image_minimax_h3_image_edit) load the INT8
ConvRot DiT and text encoder, the FP16 VAE and the W4A8 prompt expander, and
sample with KSampler at 6 steps, CFG 1, euler and simple on a zeroed negative,
with the expander behind an If/Else switch. The shipped graphs must keep each
of those, with the DGX Monarch loader and sampler in place of the stock ones.
The expander chain follows ComfyUI's: Replace Text on `{input text}` and
Generate Text, with this project's settings for it. The caption is read by
pattern, where ComfyUI reads strict JSON. The testing graph walks the checkpoint's own sigma grid
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

import importlib.util
import json
import re
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

# The widget canary holds the table of expander answers and the caption each must give.
_SPEC = importlib.util.spec_from_file_location(
    "dgxm_template_widget_canary_h3i", ROOT / "tests" / "canary" / "template_widget_canary.py")
assert _SPEC and _SPEC.loader
canary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(canary)

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
    read_id, read = _one(graph, "RegexExtract")
    marks_id, marks = _one(graph, "RegexReplace")
    expand_id, expand = _one(graph, "TextGenerateV2")
    splice_id, splice = _one(graph, "StringReplace")
    # Three If/Else switches, read back from the conditioning node's prompt: the one
    # a reader sets, the empty-request skip and the empty-caption fallback.
    switches = {key: node["inputs"] for key, node in graph.items()
                if node["class_type"] == "ComfySwitchNode"}
    assert len(switches) == 3
    switch_id = cond["prompt"][0]
    switch = switches[switch_id]
    skip_id = switch["on_true"][0]
    skip = switches[skip_id]
    fallback_id = skip["on_false"][0]
    fallback = switches[fallback_id]
    assert {switch_id, skip_id, fallback_id} == set(switches)
    # The caption: Extract Text over the expander's answer, then Replace Text (Regex).
    assert fallback["on_false"] == [marks_id, 0]
    assert marks["string"] == [read_id, 0] and read["string"] == [expand_id, 0]
    assert expand["prompt"] == [splice_id, 0]
    request_id, task_id, system_id = (
        splice["replace"][0], splice["string"][0], expand["system_prompt"][0])
    assert switch["on_false"] == [request_id, 0]
    texts = {key: node["inputs"]["value"] for key, node in graph.items()
             if node["class_type"] == "PrimitiveStringMultiline"}
    assert set(texts) == {request_id, task_id, system_id} and len(texts) == 3
    return {"cond": cond, "switch": switch, "read": read, "marks": marks, "expand": expand,
            "splice": splice, "request": texts[request_id], "task": texts[task_id],
            "system": texts[system_id], "switch_id": switch_id, "task_id": task_id,
            "system_id": system_id, "request_id": request_id, "fallback": fallback,
            "fallback_id": fallback_id, "marks_id": marks_id, "read_id": read_id,
            "expand_id": expand_id, "skip": skip, "skip_id": skip_id}


@pytest.mark.parametrize("name", [T2I, EDIT, GRID])
def test_the_expander_chain_is_wired_as_comfyui_wires_it(name):
    graph = _graph(name)
    chain = _chain(graph)
    edit = name == EDIT
    # Replace Text puts the request where the task prompt says {input text};
    # ComfyUI's templates store ["", "{input text}", ""] with both ends wired.
    assert chain["splice"]["find"] == gen_templates.H3I_MARKER == "{input text}"
    assert "expanded_caption" in chain["read"]["regex_pattern"]
    expand = chain["expand"]
    assert graph[expand["clip"][0]]["inputs"] == {"clip_name": EXPANDER, "type": "minimax",
                                                  "device": "default"}
    # This project's settings for the expander, the same in every graph.
    assert (expand["max_length"], expand["thinking"], expand["use_default_template"],
            expand["mtp"]) == (1024, False, True, "off")
    assert {key: value for key, value in expand.items()
            if key.startswith("sampling_mode")} == gen_templates.H3I_DECODING
    assert ("images.image1" in expand) is edit
    # Preview as Text shows the switch's choice from beside the path.
    _, preview = _one(graph, "PreviewAny")
    assert preview["source"] == [chain["switch_id"], 0]
    assert chain["request"] == (gen_templates.H3I_EDIT_REQUEST if edit
                                else gen_templates.H3I_REQUEST)


# The five values round 3 of the expander texts was written for, and the widgets
# that stay as ComfyUI's templates have them.
ROUND3 = {"thinking": False, "max_length": 1024, "mtp": "off",
          "sampling_mode.temperature": 0.7, "sampling_mode.top_p": 0.8}
UNCHANGED = {"sampling_mode": "on", "sampling_mode.top_k": 20, "sampling_mode.min_p": 0.0,
             "sampling_mode.repetition_penalty": 1.0, "sampling_mode.presence_penalty": 0.0,
             "sampling_mode.seed": 0, "use_default_template": True}


@pytest.mark.parametrize("name", [T2I, EDIT])
def test_generate_text_holds_the_five_values_the_texts_were_written_for(name):
    """Reasoning off, at most 1024 new tokens, mtp off, temperature 0.7 and top_p 0.8,
    in both shipped graphs alike; every other widget keeps ComfyUI's value."""
    _, expand = _one(_graph(name), "TextGenerateV2")
    assert {key: expand[key] for key in ROUND3} == ROUND3
    assert {key: expand[key] for key in UNCHANGED} == UNCHANGED
    assert expand["prompt"] != "" and isinstance(expand["prompt"], list)
    widgets = {key for key, value in expand.items() if not isinstance(value, list)}
    assert widgets == set(ROUND3) | set(UNCHANGED)
    # The generator holds them in two constants, and the UI graph stores thirteen slots.
    assert {**gen_templates.H3I_DECODING, **gen_templates.H3I_GENERATE} == {**ROUND3, **UNCHANGED}
    (stored,) = [node["widgets_values"] for node in _doc(name)["nodes"]
                 if node["type"] == "TextGenerateV2"]
    assert stored == ["", 1024, "on", 0.7, 20, 0.8, 0, 1, 0, 0, False, True, "off"]


def test_generate_texts_children_follow_the_options_order():
    children = gen_templates.DYNAMIC_WIDGETS["TextGenerateV2"]["sampling_mode"]
    assert [f"sampling_mode.{child}" for child in children["on"]] == [
        key for key in gen_templates.H3I_DECODING if key != "sampling_mode"]
    # The testing graph runs no expander; its node holds the same values all the same.
    stored = {name: [node["widgets_values"] for node in _doc(name)["nodes"]
                     if node["type"] == "TextGenerateV2"] for name in PATHS}
    assert stored[T2I] == stored[EDIT] == stored[GRID]


PROMPTS = ROOT / "tools" / "expander_prompts" / "minimax_h3_image"
# The texts as the prompt author's last round left them, by SHA-256.
PROMPT_SHA256 = {
    "system_prompt.txt": "03c7a8f5b49324f57bae53601b2084a31c695469ad3106bad7fb6f67f6c14c25",
    "t2i_task_prompt.txt": "e3f9cc2cb7b472bbd4c74ed8fdfeea92cbe031799c55a869aa7187d3551448e9",
    "edit_task_prompt.txt": "02e3bbd5bb8c0039d4eb31933ddc300707bb4665b594e22b615778b1804538f2",
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
        # The answer's key is the one the caption pattern reads.
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
    assert titles == {gen_templates.H3I_SWITCH_TITLE, gen_templates.H3I_FALLBACK_TITLE,
                      gen_templates.H3I_SKIP_TITLE}
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
def test_the_caption_is_read_by_pattern_and_the_request_is_sent_when_there_is_none(name):
    """Extract Text reads the value of the answer's outermost wrapper, so a double
    quote or a JSON object inside the caption neither loses it nor cuts it short.
    Replace Text (Regex) decodes the escapes a model writes. When the read holds no
    caption, the request goes on as written. No strict JSON read is in the graph."""
    graph = _graph(name)
    chain = _chain(graph)
    assert not [node for node in graph.values() if node["class_type"] == "JsonExtractString"]
    trims = {key: node["inputs"] for key, node in graph.items() if node["class_type"] == "StringTrim"}
    compares = {key: node["inputs"] for key, node in graph.items()
                if node["class_type"] == "StringCompare"}
    assert len(trims) == len(compares) == 2
    for inputs in trims.values():
        assert inputs["mode"] == "Both"
    for inputs in compares.values():
        assert (inputs["string_b"], inputs["mode"], inputs["case_sensitive"]) == ("", "Equal", True)

    def empty_test(switch: dict) -> list:
        """The node a switch's choice tests for emptiness, through Compare Text and Trim Text."""
        compare = compares[switch["switch"][0]]
        return trims[compare["string_a"][0]]["string"]

    assert chain["read"] == {"string": [chain["expand_id"], 0],
                             "regex_pattern": gen_templates.H3I_CAPTION_PATTERN,
                             "mode": "First Group", "case_insensitive": False,
                             "multiline": False, "dotall": True, "group_index": 2}
    assert chain["marks"] == {"string": [chain["read_id"], 0],
                              "regex_pattern": gen_templates.H3I_UNESCAPE[0],
                              "replace": gen_templates.H3I_UNESCAPE[1],
                              "case_insensitive": False, "multiline": False, "dotall": False,
                              "count": 0}
    # The fallback: the request when the caption read is empty.
    assert empty_test(chain["fallback"]) == [chain["marks_id"], 0]
    assert chain["fallback"]["on_true"] == [chain["request_id"], 0]
    assert chain["fallback"]["on_false"] == [chain["marks_id"], 0]
    # An empty request skips all of it.
    assert empty_test(chain["skip"]) == [chain["request_id"], 0]
    assert chain["skip"]["on_true"] == [chain["request_id"], 0]
    assert chain["skip"]["on_false"] == [chain["fallback_id"], 0]
    assert chain["switch"]["on_true"] == [chain["skip_id"], 0]
    # Preview as Text reads the outer switch, so it shows the text that is rendered:
    # the request with the switch false, for an empty request or after a fallback,
    # the caption otherwise.
    _, preview = _one(graph, "PreviewAny")
    assert preview["source"] == chain["cond"]["prompt"] == [chain["switch_id"], 0]


def _flags(case_insensitive: bool, multiline: bool, dotall: bool) -> int:
    return ((re.IGNORECASE if case_insensitive else 0) | (re.MULTILINE if multiline else 0)
            | (re.DOTALL if dotall else 0))


def _regex_extract(text: str, pattern: str, mode: str, case_insensitive: bool,
                   multiline: bool, dotall: bool, group_index: int) -> str:
    """Extract Text in First Group mode, as ComfyUI 0.45.0 runs it
    (comfy_extras/nodes_string.py)."""
    assert mode == "First Group"
    match = re.search(pattern, text, _flags(case_insensitive, multiline, dotall))
    return match.group(group_index) if match and len(match.groups()) >= group_index else ""


def _regex_replace(text: str, pattern: str, replace: str, case_insensitive: bool,
                   multiline: bool, dotall: bool, count: int) -> str:
    """Replace Text (Regex), from the same file."""
    return re.sub(pattern, replace, text, count=count,
                  flags=_flags(case_insensitive, multiline, dotall))


def _trim(text: str, mode: str) -> str:
    assert mode == "Both"
    return text.strip()


def _compare(left: str, right: str, mode: str, case_sensitive: bool) -> bool:
    assert mode == "Equal" and case_sensitive is True
    return left == right


STAND_INS = {"RegexExtract": _regex_extract, "RegexReplace": _regex_replace,
             "StringTrim": _trim, "StringCompare": _compare}


def _stand_in(kind: str, *args):
    return STAND_INS[kind](*args)


@pytest.mark.parametrize("name", [T2I, EDIT, GRID])
def test_the_pattern_holds_against_the_table_of_answers(name):
    """Each answer in the canary's table gives the caption the table states, with the
    widget values this graph holds. The four nodes are stand-ins written from
    ComfyUI's code; the widget canary runs the same table through ComfyUI's own."""
    settings = canary.parse_settings(_graph(name))
    assert settings == {
        "read": (gen_templates.H3I_CAPTION_PATTERN, "First Group", False, False, True, 2),
        "marks": (*gen_templates.H3I_UNESCAPE, False, False, False, 0)}
    assert canary.parse_chain_errors(settings, _stand_in) == []
    counts = {read: sum(1 for row in canary.PARSE_CASES if row[3] == read)
              for read in ("read", "request")}
    assert counts == {"read": 37, "request": 10} and len(canary.PARSE_CASES) == 47
    # The table has teeth. A pattern that stops at the first closing quote loses the
    # captions with a quote inside them, one that runs to the last quote and brace
    # takes a second key into the caption, and a read without the decode pass leaves
    # the escapes in.
    naive = {**settings, "read": (r'"expanded_caption"\s*:\s*"([^"]*)"', "First Group",
                                  False, False, True, 1)}
    assert len(canary.parse_chain_errors(naive, _stand_in)) >= 8
    greedy = {**settings, "read": (r'"expanded_caption\\?"\s*:\s*(\\)?"(.*)(?(1)\\"\s*"?|")\s*\}',
                                   "First Group", False, False, True, 2)}
    assert len(canary.parse_chain_errors(greedy, _stand_in)) >= 4
    raw = {**settings, "marks": (r"(?!)", "", False, False, False, 0)}
    assert len(canary.parse_chain_errors(raw, _stand_in)) >= 5


def test_the_strict_read_this_replaces_was_wrong_in_the_shapes_the_texts_invite():
    """Extract Text from JSON, as ComfyUI's own templates read the caption: the key
    of the first object that parses. For a caption with a raw double quote it finds
    no object; for a copied object that holds the answer key it returns that inner
    value; for a raw quote followed by a closing brace it returns a cut caption. The
    table gives the whole caption for each."""
    def strict(text: str) -> str:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError:
                continue
            value = data.get("expanded_caption")
            return "" if value is None else str(value)
        return ""

    rows = {what: (answer, caption) for what, answer, caption, _read in canary.PARSE_CASES}
    answer, caption = rows["inner double quotes"]
    assert strict(answer) == "" and caption
    answer, caption = rows["an inner object that itself holds the answer key"]
    assert strict(answer) == "hello" != caption
    answer, caption = rows["a raw double quote then a closing brace inside the caption"]
    assert strict(answer) == "A sign with the marks " and caption.startswith(strict(answer))
    # Where the strict read is right, the pattern read agrees with it byte for byte.
    for what in ("a valid object", "escaped double quotes", "an escaped backslash",
                 "an escaped path", "an escaped slash", "a code fence",
                 "text before and after the object",
                 "an escaped double quote at the caption's end"):
        answer, caption = rows[what]
        assert strict(answer) == caption, what


def _rendered(name: str, request: str, answer: str) -> tuple[str, int]:
    """The text the conditioning node receives with the outer switch true, when the
    expander's answer is ``answer``, and how many times Generate Text ran.

    A CPU walk of the text path with stand-ins for the stock nodes, each as ComfyUI
    0.45.0's code reads (comfy_extras/nodes_string.py, nodes_logic.py). If/Else is
    lazy there: it asks only for the branch its switch selects, so the walk
    evaluates only that branch. Generate Text is the one node not run: its output is
    the answer given, and each visit is counted.
    """
    graph = _graph(name)
    chain = _chain(graph)
    graph[chain["request_id"]]["inputs"]["value"] = request
    chain["switch"]["switch"] = True
    runs: list[str] = []
    done: dict[str, object] = {}

    def value(ref):
        if not isinstance(ref, list):
            return ref
        if ref[0] not in done:  # ComfyUI runs a node once a queue and keeps its output
            done[ref[0]] = run(ref[0])
        return done[ref[0]]

    def run(key: str):
        kind, inputs = graph[key]["class_type"], graph[key]["inputs"]
        if kind == "PrimitiveStringMultiline":
            return inputs["value"]
        if kind == "StringReplace":
            return value(inputs["string"]).replace(inputs["find"], value(inputs["replace"]))
        if kind == "TextGenerateV2":
            assert value(inputs["prompt"]).strip(), "the expander was handed an empty prompt"
            runs.append(key)
            return answer
        if kind == "RegexReplace":
            return _regex_replace(value(inputs["string"]), inputs["regex_pattern"], inputs["replace"],
                                  inputs["case_insensitive"], inputs["multiline"],
                                  inputs["dotall"], inputs["count"])
        if kind == "RegexExtract":
            return _regex_extract(value(inputs["string"]), inputs["regex_pattern"], inputs["mode"],
                                  inputs["case_insensitive"], inputs["multiline"],
                                  inputs["dotall"], inputs["group_index"])
        if kind == "StringTrim":
            return _trim(value(inputs["string"]), inputs["mode"])
        if kind == "StringCompare":
            return _compare(value(inputs["string_a"]), value(inputs["string_b"]),
                            inputs["mode"], inputs["case_sensitive"])
        if kind == "ComfySwitchNode":
            return value(inputs["on_true"] if value(inputs["switch"]) else inputs["on_false"])
        raise AssertionError(f"{kind} is not on the text path")

    return value(chain["cond"]["prompt"]), len(runs)


CAPTION = "A photograph of a small weather station at first light, frost on the grass."


@pytest.mark.parametrize("name", [T2I, EDIT, GRID])
def test_an_empty_request_never_starts_a_language_model_run(name):
    """An empty request, or one of spaces and line breaks, goes past Generate Text:
    the model gets the request as written, as with the expander off, and the
    expander is never asked for an answer."""
    for request in ("", "  ", " \n\t "):
        assert _rendered(name, request, json.dumps({"expanded_caption": CAPTION})) == (request, 0)
    if name != GRID:  # the testing graph holds no task prompt to expand a request with
        assert _rendered(name, "a boat", json.dumps({"expanded_caption": CAPTION})) == (CAPTION, 1)


@pytest.mark.parametrize(("name", "asked"), [(T2I, "a boat on a beach"),
                                               (EDIT, "make <Picture 1> snowy")])
def test_the_whole_text_path_gives_the_model_a_caption_or_the_request(name, asked):
    """Every answer in the table, through the graph's own wiring: a caption the read
    finds goes to the model as read, and with none the request does."""
    for what, answer, caption, read in canary.PARSE_CASES:
        expected = asked if read == "request" else caption
        assert _rendered(name, asked, answer) == (expected, 1), what
    # A caption goes through untouched, untrimmed.
    padded = json.dumps({"expanded_caption": f" {CAPTION} "})
    assert _rendered(name, asked, padded) == (f" {CAPTION} ", 1)


def test_no_text_outside_the_note_and_the_two_prompt_boxes_is_long():
    """ComfyUI's text to image task prompt runs to 15,823 characters, its edit one
    is long too, and its demo caption runs to 3,308. This project's own texts run to
    947, 3,164 and 3,701 bytes, and no other widget comes near."""
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
    # What turning it on costs: the expander's memory, the tokens it may write, and
    # the wait a first hardware probe timed for this graph.
    tokens = 1024
    seconds = 13 if edit else 7
    assert gen_templates.H3I_PROBE_SECONDS == {"t2i": 7, "edit": 13}
    assert (f"the driver loads the 7.3 GB expander beside the text encoder, and Generate "
            f"Text may write up to {tokens} tokens before the render starts, a wait of "
            f"about {seconds} seconds on a DGX Spark in a first probe") in note
    assert "not yet timed" not in note
    assert "Generate Text runs the expander model on the driver, with reasoning off" in note
    # A user may replace the texts, and what a replacement must keep.
    assert ("You may replace either text with your own: keep {input text} in the task "
            "prompt where the request belongs, and have the answer hold the caption under "
            "the key expanded_caption") in note
    # The fallback, and that the preview shows what is rendered.
    assert ("Extract Text reads the caption by pattern, not as strict JSON, so a double "
            "quote or a JSON object inside the caption does not lose it or cut it short; "
            "the texts ask the expander to copy drawn words exactly, and they expect this "
            f"read. If the read holds no caption, the switch titled "
            f"{gen_templates.H3I_FALLBACK_TITLE} sends the request as written instead. An "
            "empty request skips the expander: no language model runs for it. Preview as "
            "Text shows the text that is rendered either way") in note
    # The refusal, and what an expander-on render is worth as a comparison.
    assert ("refuses a graph whose expander is on with an empty task prompt, or with one "
            "that lacks the marker "
            f"(docs/TROUBLESHOOTING.md #{expander_prompt.TROUBLESHOOTING})") in note
    assert ("text a language model sampled, at Generate Text's own seed and not the "
            "sampler's") in note
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
