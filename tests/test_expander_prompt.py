"""A prompt expander that cannot see the request refuses before the DiT loads or a picture renders.

The MiniMax-H3-Image templates wire ComfyUI's expander chain and ship it off,
with the task prompt box empty: the text ComfyUI's own templates carry there is
MiniMax's, and this repository holds none of it. Replace Text over an empty
task prompt is an empty text, so with the switch on Generate Text answers an
empty prompt, Extract Text from JSON finds no caption, and the model renders
from whatever came out with no error anywhere. ComfyUI raises nothing for
that, so a DGX Monarch loader, sampler or Identity Gate reads the chain in the
prompt graph and refuses, class P.

These tests hold the sentence, where it answers, and that it answers nothing
else: a graph with the switch off, a filled task prompt, an expander wired
some other way, a switch whose choice the graph does not state, and every
other committed template. CPU only: the graphs are read as JSON, and stubs
stand in for ComfyUI and the fleet.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import (
    expander_prompt,
    gate,
    gate_node,
    graph_live,
    loader_preflight,
    loaders,
    render_validation,
    requeue_price,
    samplers,
)
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sweep import convert  # noqa: E402

T2I = "dgx-monarch-minimax-h3-image-t2i"
EDIT = "dgx-monarch-minimax-h3-image-edit"
GRID = "dgx-monarch-test-minimax-h3-image-grid"
FAMILY_GRAPHS = (T2I, EDIT, GRID)
TEMPLATES = {path.stem: path for path in (
    sorted((ROOT / "example_workflows").glob("*.json"))
    + sorted((ROOT / "tests" / "fixtures" / "workflows" / "generated").glob("*.json")))}
DIT = "minimax_h3_image_pruned_int8_convrot.safetensors"
# A task prompt of this test's own, with the marker where the request belongs.
FILLED = "Rewrite the request as one caption.\ninput_text: {input text}\noutput_text:"
# The same without the marker: MiniMax's own file ends its text to image task
# prompt with a bare pair of braces, so a paste that skips the Note's last step
# looks like this.
UNMARKED = "Rewrite the request as one caption.\ninput_text: {}\noutput_text:"


def _graph(name: str) -> dict:
    doc = json.loads(TEMPLATES[name].read_text(encoding="utf-8"))
    return convert.convert(doc, convert.repo_widget_names(ROOT))


def _one(graph: dict, class_type: str) -> tuple[str, dict]:
    ((key, node),) = [(key, node) for key, node in graph.items()
                      if node["class_type"] == class_type]
    return key, node["inputs"]


def _switched_on(name: str, task: str | None = None) -> dict:
    """The shipped graph with its If/Else switch on and, when given, a task prompt."""
    graph = _graph(name)
    _, switch = _one(graph, "ComfySwitchNode")
    switch["switch"] = True
    if task is not None:
        _, splice = _one(graph, "StringReplace")
        graph[splice["string"][0]]["inputs"]["value"] = task
    return graph


def _empty_sentence(graph: dict) -> str:
    splice_id, _ = _one(graph, "StringReplace")
    expand_id, _ = _one(graph, "TextGenerateV2")
    return expander_prompt.EMPTY_TASK_PROMPT.format(replace=splice_id, generate=expand_id)


# ---- What it answers -----------------------------------------------------------

@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_the_shipped_graph_refuses_nothing(name):
    graph = _graph(name)
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_the_switch_on_over_the_empty_task_prompt_refuses_class_p(name):
    graph = _switched_on(name)
    sentence = expander_prompt.dead_expander(graph)
    assert sentence == _empty_sentence(graph)
    with pytest.raises(UnsupportedModelError) as raised:
        expander_prompt.refuse_dead_expander(graph)
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False
    assert sentence in text
    assert text.endswith(f"See docs/TROUBLESHOOTING.md #{expander_prompt.TROUBLESHOOTING}.")
    # What happened, that nothing rendered, and both ways on.
    assert "The prompt expander is on and its task prompt is empty." in text
    assert "No picture was rendered." in text
    assert "Fill the text box wired to Replace Text with the task prompt" in text
    assert "set the If/Else Switch to false" in text
    assert text.isascii() and len(text) < 1000


def test_a_task_prompt_of_spaces_and_line_breaks_is_empty_too():
    graph = _switched_on(T2I, " \n\t ")
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_a_filled_task_prompt_that_holds_the_marker_passes(name):
    graph = _switched_on(name, FILLED)
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


def test_a_task_prompt_without_the_marker_refuses_and_names_the_marker():
    graph = _switched_on(T2I, UNMARKED)
    splice_id, _ = _one(graph, "StringReplace")
    expand_id, _ = _one(graph, "TextGenerateV2")
    sentence = expander_prompt.dead_expander(graph)
    assert sentence == expander_prompt.MISSING_MARKER.format(
        replace=splice_id, generate=expand_id, find="{input text}")
    assert "holds '{input text}'" in sentence and "ignores the request" in sentence
    with pytest.raises(UnsupportedModelError) as raised:
        expander_prompt.refuse_dead_expander(graph)
    tag = parse_leading_refusal_tag(str(raised.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS


def test_a_typed_task_prompt_on_replace_text_is_read_as_a_wired_one_is():
    """Replace Text's own `string` box, with no text box wired to it."""
    graph = _switched_on(T2I)
    _, splice = _one(graph, "StringReplace")
    splice["string"] = ""
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)
    splice["string"] = FILLED
    assert expander_prompt.dead_expander(graph) is None
    splice["string"] = UNMARKED
    assert "the request cannot reach it" in (expander_prompt.dead_expander(graph) or "")
    # A typed `replace` is no request to lose, so the marker rule stays quiet.
    splice["replace"] = "a fixed word"
    assert expander_prompt.dead_expander(graph) is None


# ---- What it leaves alone --------------------------------------------------------

def test_only_what_the_graph_states_is_read():
    # A task prompt that comes from a node this cannot read.
    graph = _switched_on(T2I)
    _, splice = _one(graph, "StringReplace")
    graph[splice["string"][0]]["class_type"] = "SomeTextFileLoader"
    assert expander_prompt.dead_expander(graph) is None
    # Generate Text fed by something other than Replace Text.
    graph = _switched_on(T2I)
    _, expand = _one(graph, "TextGenerateV2")
    _, switch = _one(graph, "ComfySwitchNode")
    expand["prompt"] = list(switch["on_false"])
    assert expander_prompt.dead_expander(graph) is None
    # Generate Text with a typed prompt.
    expand["prompt"] = ""
    assert expander_prompt.dead_expander(graph) is None


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_a_switch_driven_by_a_boolean_box_is_read_as_that_box(name):
    """ComfyUI's own edit template drives its If/Else switch from a Boolean box."""
    graph = _graph(name)
    _, switch = _one(graph, "ComfySwitchNode")
    graph["900"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": True}}
    switch["switch"] = ["900", 0]
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)
    graph["900"]["inputs"]["value"] = False
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


def test_a_switch_whose_choice_the_graph_does_not_state_refuses_nothing():
    """The live walk keeps both branches of such a switch, right for a price; the
    expander may well be off, and a refusal there would stop a good render."""
    # Driven by a node this cannot read.
    graph = _switched_on(T2I)
    _, switch = _one(graph, "ComfySwitchNode")
    graph["900"] = {"class_type": "SomeBooleanMaker", "inputs": {}}
    switch["switch"] = ["900", 0]
    assert expander_prompt.dead_expander(graph) is None
    # Driven by a Boolean box whose own value is wired.
    graph["900"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": ["901", 0]}}
    graph["901"] = {"class_type": "SomeBooleanMaker", "inputs": {}}
    assert expander_prompt.dead_expander(graph) is None
    # One branch left open: ComfyUI may evaluate either.
    graph = _switched_on(T2I)
    _, switch = _one(graph, "ComfySwitchNode")
    del switch["on_false"]
    assert expander_prompt.dead_expander(graph) is None
    # A second switch elsewhere in the running graph that states no choice.
    graph = _switched_on(T2I)
    _, save = _one(graph, "SaveImage")
    graph["900"] = {"class_type": "ComfySoftSwitchNode",
                    "inputs": {"switch": ["901", 0], "on_true": list(save["images"]),
                               "on_false": list(save["images"])}}
    graph["901"] = {"class_type": "SomeBooleanMaker", "inputs": {}}
    save["images"] = ["900", 0]
    assert expander_prompt.dead_expander(graph) is None
    # ComfyUI's Branch node is lazy too, and nothing here reads which branch it runs.
    graph = _switched_on(T2I)
    extract_id, _ = _one(graph, "JsonExtractString")
    graph["900"] = {"class_type": "BranchNode", "inputs": {"branch0": [extract_id, 0]}}
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
@pytest.mark.parametrize("tap", ["extract", "thinking", "replace"])
def test_an_expander_that_runs_only_for_a_preview_refuses_nothing(name, tap):
    """The switch is false and the boxes are empty, as shipped, and one more Preview
    as Text hangs on the chain. ComfyUI runs the expander for that preview alone; the
    picture takes the request as written, so there is nothing to refuse."""
    graph = _graph(name)
    extract_id, _ = _one(graph, "JsonExtractString")
    expand_id, _ = _one(graph, "TextGenerateV2")
    splice_id, _ = _one(graph, "StringReplace")
    source = {"extract": [extract_id, 0], "thinking": [expand_id, 1], "replace": [splice_id, 0]}[tap]
    graph["900"] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None
    # The same graph with the switch on feeds the render from the expander, and refuses.
    _, switch = _one(graph, "ComfySwitchNode")
    switch["switch"] = True
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_comfyuis_layout_with_the_preview_on_the_text_path_still_refuses(name):
    """ComfyUI's templates wire switch, then Preview as Text, then the conditioning
    prompt. Preview as Text hands its string on, so the render still reads the chain."""
    graph = _graph(name)
    switch_id, switch = _one(graph, "ComfySwitchNode")
    preview_id, _ = _one(graph, "PreviewAny")
    _, cond = _one(graph, "MiniMaxH3ImageConditioning")
    assert cond["prompt"] == [switch_id, 0]
    cond["prompt"] = [preview_id, 0]
    assert expander_prompt.dead_expander(graph) is None
    switch["switch"] = True
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


def test_a_graph_with_no_dgx_monarch_render_refuses_nothing():
    """Stock samplers render nothing through this pack, so their graphs are not read."""
    graph = _switched_on(T2I)
    sampler_id, _ = _one(graph, "DGXMonarchKSampler")
    graph[sampler_id]["class_type"] = "KSampler"
    assert expander_prompt.dead_expander(graph) is None


def test_the_render_classes_are_the_nodes_that_call_the_refusal():
    from dgx_monarch import nodes as pack

    calling = set()
    for class_name, cls in pack.NODE_CLASS_MAPPINGS.items():
        method = getattr(cls, getattr(cls, "FUNCTION", ""), None)
        code = getattr(getattr(method, "__wrapped__", method), "__code__", None)
        if code is not None and "refuse_dead_expander" in code.co_names:
            calling.add(class_name)
    # The two UNET loaders call it too, and render nothing themselves.
    loaders = {"DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader"}
    assert calling - loaders == set(expander_prompt.RENDER_CLASSES)
    assert loaders <= calling


def test_the_refusal_and_every_link_name_the_expander_entry():
    """A merge once left this number on another entry: the pointer test passes on any
    number that exists, so the heading and the links are held to the constant here."""
    number = expander_prompt.TROUBLESHOOTING
    entries = (ROOT / "docs" / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
    assert (f"\n## {number}. A render refuses because the prompt expander is on "
            "with an empty task prompt\n") in entries
    slug = f"#{number}-a-render-refuses-because-the-prompt-expander-is-on-with-an-empty-task-prompt"
    for rel, count in (("CHANGELOG.md", 1), ("docs/MODELS.md", 1), ("docs/SWEEP.md", 1),
                       ("docs/VALIDATION.md", 2)):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert text.count(slug) == count, rel
        assert "-a-render-refuses-because-the-prompt-expander" not in text.replace(slug, ""), rel
    for name in (T2I, EDIT):
        doc = json.loads(TEMPLATES[name].read_text(encoding="utf-8"))
        (note,) = [node["widgets_values"][0] for node in doc["nodes"] if node["type"] == "Note"]
        assert f"(docs/TROUBLESHOOTING.md #{number})" in note


@pytest.mark.parametrize("prompt", [None, "", [], 7, {"1": "not a node"}, {"1": {}},
                                    {"1": {"class_type": "TextGenerateV2"}},
                                    {"1": {"class_type": "TextGenerateV2",
                                           "inputs": {"prompt": ["9", 0]}}}])
def test_a_graph_it_cannot_read_refuses_nothing(prompt):
    assert expander_prompt.dead_expander(prompt) is None
    assert expander_prompt.refuse_dead_expander(prompt) is None


def test_a_read_that_fails_refuses_nothing(monkeypatch):
    def broken(_prompt):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(graph_live, "live_nodes", broken)
    with pytest.raises(RuntimeError):
        expander_prompt.dead_expander(_switched_on(T2I))
    assert expander_prompt.refuse_dead_expander(_switched_on(T2I)) is None


def test_no_other_committed_template_meets_the_refusal_as_shipped_or_switched_on():
    """Every committed graph as shipped, and again with each If/Else switch on."""
    assert len(TEMPLATES) > 100
    refused = []
    for name in TEMPLATES:
        graph = _graph(name)
        assert expander_prompt.dead_expander(graph) is None, name
        switches = [node for node in graph.values() if node["class_type"] == "ComfySwitchNode"]
        for node in switches:
            node["inputs"]["switch"] = True
        if switches and expander_prompt.dead_expander(graph) is not None:
            refused.append(name)
    assert sorted(refused) == sorted(FAMILY_GRAPHS)


# ---- Where it answers ------------------------------------------------------------

def _explode(reached: list, what: str):
    def explode(*_args, **_kwargs):
        reached.append(what)
        raise AssertionError(f"{what} ran before the expander refusal")
    return explode


def _model():
    mesh = types.SimpleNamespace(topology_preset="auto", handle=None, worker_args={},
                                 attention=None)
    return types.SimpleNamespace(mesh=mesh, unet_name=DIT, options={})


def test_the_loader_refuses_before_any_price_heal_or_load(monkeypatch):
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, _name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    reached: list = []
    monkeypatch.setattr(render_validation, "preflight_graph_batch_divides_dp",
                        _explode(reached, "the batch preflight"))
    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology",
                        _explode(reached, "the policy card"))
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        _explode(reached, "the footprint card"))
    monkeypatch.setattr(loaders, "ensure_live", _explode(reached, "ensure_live"))
    graph = _switched_on(T2I)
    with pytest.raises(UnsupportedModelError) as raised:
        loaders.DGXMonarchUNETLoader().load(_model().mesh, DIT, prompt=graph)
    assert _empty_sentence(graph) in str(raised.value) and reached == []


SAMPLER_CALLS = {
    "DGXMonarchKSampler": lambda model, graph: samplers.DGXMonarchKSampler().sample(
        model, 42, 6, 1.0, "euler", "simple", [], [], {"samples": None}, prompt=graph),
    "DGXMonarchKSamplerPipeline": lambda model, graph: samplers.DGXMonarchKSamplerPipeline().sample(
        model, "42", 6, 1.0, "euler", "simple", [], [], {"samples": None}, prompt=graph),
    "DGXMonarchKSamplerAdvanced": lambda model, graph: samplers.DGXMonarchKSamplerAdvanced().sample(
        model, "enable", 42, 6, 1.0, "euler", "simple", [], [], {"samples": None}, 0, 6,
        "disable", prompt=graph),
    "DGXMonarchSamplerCustom": lambda model, graph: samplers.DGXMonarchSamplerCustom().sample(
        types.SimpleNamespace(seed=42), {"model": model}, object(), None, {"samples": None},
        prompt=graph),
}


@pytest.mark.parametrize("node", sorted(SAMPLER_CALLS))
def test_every_sampler_refuses_before_its_price_and_its_render(monkeypatch, node):
    """A re-queue keeps the loader's output, so the loader's refusal never runs for it."""
    reached: list = []
    monkeypatch.setattr(requeue_price, "price_cached_loader", _explode(reached, "the price"))
    monkeypatch.setattr(samplers, "run_render", _explode(reached, "the render"))
    graph = _switched_on(GRID if node == "DGXMonarchSamplerCustom" else T2I)
    with pytest.raises(UnsupportedModelError) as raised:
        SAMPLER_CALLS[node](_model(), graph)
    assert _empty_sentence(graph) in str(raised.value) and reached == []


def test_every_sampling_node_of_the_pack_is_one_of_those_four():
    """A sampler added later must refuse too: this list is the pack's render sites."""
    source = (ROOT / "src" / "dgx_monarch" / "nodes" / "samplers.py").read_text(encoding="utf-8")
    assert source.count("requeue_price.price_cached_loader(") == len(SAMPLER_CALLS)
    assert source.count("expander_prompt.refuse_dead_expander(prompt)") == len(SAMPLER_CALLS)


def test_identity_gate_refuses_before_its_price_and_its_ceremony(monkeypatch):
    reached: list = []
    monkeypatch.setattr(requeue_price, "price_cached_loader", _explode(reached, "the price"))
    monkeypatch.setattr(gate, "run_identity_ceremony", _explode(reached, "the ceremony"))
    monkeypatch.setattr(gate, "conditioning_for_wire", lambda conds: conds)
    graph = _switched_on(T2I)
    with pytest.raises(UnsupportedModelError) as raised:
        gate_node.DGXMonarchIdentityGate().gate(
            _model(), [], [], {"samples": None}, 42, 6, 1.0, "euler", "simple", prompt=graph)
    assert _empty_sentence(graph) in str(raised.value) and reached == []


@pytest.mark.parametrize("node", sorted(SAMPLER_CALLS))
def test_a_sampler_reaches_its_price_when_the_graph_is_as_shipped(monkeypatch, node):
    class Priced(Exception):
        pass

    def priced(*_args, **_kwargs):
        raise Priced

    monkeypatch.setattr(requeue_price, "price_cached_loader", priced)
    graph = _graph(GRID if node == "DGXMonarchSamplerCustom" else T2I)
    with pytest.raises(Priced):
        SAMPLER_CALLS[node](_model(), graph)
