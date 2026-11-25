"""A prompt expander that cannot see the request refuses before the DiT loads or a picture renders.

The MiniMax-H3-Image templates wire ComfyUI's expander chain and ship it off.
The two shipped templates carry this project's own task prompt and system
prompt; the testing graph ships both boxes empty, and a user may empty or
replace a task prompt. Replace Text over an empty task prompt is an empty
text, so with the switch on Generate Text answers an empty prompt and the
render follows what comes of that, with no error anywhere. ComfyUI raises
nothing for that, so a DGX Monarch loader, sampler or Identity Gate reads the
chain in the prompt graph and refuses, class P.

These tests hold the sentence, where it answers, and that it answers nothing
else: a graph with the switch off, the shipped texts with the switch on, an
expander wired some other way or behind a switch whose choice the graph does
not state, and every other committed template. They hold that the templates'
empty-caption fallback does not quiet the refusal. CPU only: the graphs are
read as JSON, and stubs stand in for ComfyUI and the fleet.
"""
from __future__ import annotations

import inspect
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
# The same without the marker, as a replacement text that keeps a bare pair of
# braces where the marker belongs would read.
UNMARKED = "Rewrite the request as one caption.\ninput_text: {}\noutput_text:"


def _graph(name: str) -> dict:
    doc = json.loads(TEMPLATES[name].read_text(encoding="utf-8"))
    return convert.convert(doc, convert.repo_widget_names(ROOT))


def _one(graph: dict, class_type: str) -> tuple[str, dict]:
    ((key, node),) = [(key, node) for key, node in graph.items()
                      if node["class_type"] == class_type]
    return key, node["inputs"]


def _outer(graph: dict) -> tuple[str, dict]:
    """The If/Else switch a reader sets. The other one, the empty-caption fallback,
    takes its choice from Compare Text over a link."""
    ((key, inputs),) = [(key, node["inputs"]) for key, node in graph.items()
                        if node["class_type"] == "ComfySwitchNode"
                        and isinstance(node["inputs"].get("switch"), bool)]
    return key, inputs


def _with_task(name: str, task: str | None, on: bool) -> dict:
    graph = _graph(name)
    _, switch = _outer(graph)
    switch["switch"] = on
    if task is not None:
        _, splice = _one(graph, "StringReplace")
        graph[splice["string"][0]]["inputs"]["value"] = task
    return graph


def _switched_on(name: str, task: str | None = "") -> dict:
    """The graph with its switch on over an emptied task prompt box, or over ``task``;
    None keeps the box as the graph ships it."""
    return _with_task(name, task, True)


def _emptied(name: str) -> dict:
    """The graph with its switch off and its task prompt box emptied."""
    return _with_task(name, "", False)


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


@pytest.mark.parametrize("name", [T2I, EDIT])
def test_the_shipped_texts_with_the_switch_on_refuse_nothing(name):
    """The two shipped graphs hold this project's task prompts, each with the marker."""
    graph = _switched_on(name, None)
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


def test_the_testing_graph_with_the_switch_on_refuses():
    """Its boxes ship empty, so turning its expander on is the case the refusal is for."""
    graph = _switched_on(GRID, None)
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


def _switches(graph: dict) -> dict:
    """The three If/Else switches of a family graph by role, from the outer one in."""
    outer_id, outer = _outer(graph)
    skip_id = outer["on_true"][0]
    fallback_id = graph[skip_id]["inputs"]["on_false"][0]
    ids = {"outer": outer_id, "skip": skip_id, "fallback": fallback_id}
    assert all(graph[key]["class_type"] == "ComfySwitchNode" for key in ids.values())
    assert len(set(ids.values())) == 3
    return ids


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
@pytest.mark.parametrize("task", ["", UNMARKED])
def test_the_empty_caption_fallback_does_not_quiet_the_refusal(name, task):
    """The empty-caption fallback takes its choice from Compare Text over the
    expander's own answer. No graph states that choice, so neither of its branches is
    sure to run, but its `switch` input is: ComfyUI must run the expander to choose.
    With the outer switch true and a request to expand, the refusal therefore still
    decides."""
    graph = _switched_on(name, task)
    ids = _switches(graph)
    expand_id, _ = _one(graph, "TextGenerateV2")
    assert graph_live.is_link(graph[ids["fallback"]]["inputs"]["switch"])
    assert graph_live.skipped_branch(graph[ids["fallback"]]) is None
    sure = graph_live.live_nodes(graph, sure=True)
    # The skip switch is read through, since its choice is not stated on the node:
    # only the walk in expander_prompt knows the request box decides it.
    assert expand_id not in sure
    sentence = expander_prompt.dead_expander(graph)
    assert sentence is not None and sentence.startswith("The prompt expander is on and ")
    assert ("its task prompt is empty" in sentence) is (task == "")
    with pytest.raises(UnsupportedModelError):
        expander_prompt.refuse_dead_expander(graph)


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
@pytest.mark.parametrize("asked", ["", "   ", " \n\t "])
def test_an_empty_request_runs_no_expander_and_refuses_nothing(name, asked):
    """An empty request goes past Generate Text: the If/Else that skips the expander
    takes its choice from Compare Text over the trimmed request box, which the graph
    states. So the walk knows the expander does not run, and an empty task prompt,
    which would refuse with a request to expand, has nothing to refuse here."""
    graph = _switched_on(name, "")
    _, splice = _one(graph, "StringReplace")
    request_id = splice["replace"][0]
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)
    graph[request_id]["inputs"]["value"] = asked
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None
    graph[request_id]["inputs"]["value"] = "a boat"
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


def test_a_request_from_a_node_this_cannot_read_leaves_the_skip_undecided():
    """The skip of an empty request is decided from the request box. When the request
    comes from a node this cannot read, the graph no longer states whether the
    expander runs, so the check steps aside, as for any switch that states no choice."""
    graph = _switched_on(T2I, "")
    _, splice = _one(graph, "StringReplace")
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)
    graph[splice["replace"][0]]["class_type"] = "SomeTextFileLoader"
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


def test_a_switch_whose_choice_compare_text_states_is_read_as_that_choice():
    """Compare Text over texts the graph states, through Trim Text, in each mode."""
    def graph(left: str, right: str, mode: str, sensitive: bool = True, trim: str = "Both"):
        return {"1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": left}},
                "2": {"class_type": "StringTrim", "inputs": {"string": ["1", 0], "mode": trim}},
                "3": {"class_type": "StringCompare",
                      "inputs": {"string_a": ["2", 0], "string_b": right, "mode": mode,
                                 "case_sensitive": sensitive}}}

    choice = expander_prompt._choice
    assert choice(graph("  ", "", "Equal"), ["3", 0]) is True
    assert choice(graph(" a ", "", "Equal"), ["3", 0]) is False
    assert choice(graph(" a", "a", "Equal", trim="Left"), ["3", 0]) is True
    assert choice(graph(" a", "a", "Equal", trim="Right"), ["3", 0]) is False
    assert choice(graph("Boat", "bo", "Starts With"), ["3", 0]) is False
    assert choice(graph("Boat", "bo", "Starts With", sensitive=False), ["3", 0]) is True
    assert choice(graph("Boat", "at", "Ends With"), ["3", 0]) is True
    # What the graph does not state is not guessed.
    assert choice(graph("a", "", "Some New Mode"), ["3", 0]) is None
    assert choice(graph("a", "", "Equal", trim="Middle"), ["3", 0]) is None
    unknown = graph("a", "", "Equal")
    unknown["1"]["class_type"] = "SomeTextFileLoader"
    assert choice(unknown, ["3", 0]) is None
    wired = graph("a", "", "Equal")
    wired["3"]["inputs"]["string_b"] = ["9", 0]
    assert choice(wired, ["3", 0]) is None
    assert choice({}, ["3", 0]) is None and choice({}, "true") is None


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_with_the_outer_switch_false_nothing_of_the_expander_runs(name):
    """The outer switch is lazy and states false, so ComfyUI never asks for the skip,
    the fallback, the read of the caption, Generate Text or the expander's loader:
    the walk the loader prices with drops them all, and so does the sure one."""
    graph = _with_task(name, None, False)
    ids = _switches(graph)
    expand_id, expand = _one(graph, "TextGenerateV2")
    chain = {ids["skip"], ids["fallback"], expand_id, expand["clip"][0],
             *(key for key, node in graph.items() if node["class_type"] in (
                 "StringCompare", "StringTrim", "RegexExtract", "RegexReplace",
                 "StringReplace"))}
    assert len(chain) == 11
    assert not chain & graph_live.live_nodes(graph)
    assert not chain & graph_live.live_nodes(graph, sure=True)
    graph[ids["outer"]]["inputs"]["switch"] = True
    assert chain <= graph_live.live_nodes(graph)


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
    assert "set the switch titled If/Else Switch (expander on or off) to false" in text
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
    _, switch = _outer(graph)
    expand["prompt"] = list(switch["on_false"])
    assert expander_prompt.dead_expander(graph) is None
    # Generate Text with a typed prompt.
    expand["prompt"] = ""
    assert expander_prompt.dead_expander(graph) is None


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_a_switch_driven_by_a_boolean_box_is_read_as_that_box(name):
    """ComfyUI's own edit template drives its If/Else switch from a Boolean box."""
    graph = _emptied(name)
    _, switch = _outer(graph)
    graph["900"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": True}}
    switch["switch"] = ["900", 0]
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)
    graph["900"]["inputs"]["value"] = False
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


def test_an_expander_behind_a_switch_that_states_no_choice_refuses_nothing():
    """The live walk keeps both branches of such a switch, right for a price; the
    expander may well be off, and a refusal there would stop a good render. The sure
    walk follows neither branch."""
    # Driven by a node this cannot read.
    graph = _switched_on(T2I)
    _, switch = _outer(graph)
    graph["900"] = {"class_type": "SomeBooleanMaker", "inputs": {}}
    switch["switch"] = ["900", 0]
    assert expander_prompt.dead_expander(graph) is None
    # Driven by a Boolean box whose own value is wired.
    graph["900"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": ["901", 0]}}
    graph["901"] = {"class_type": "SomeBooleanMaker", "inputs": {}}
    assert expander_prompt.dead_expander(graph) is None
    # One branch left open: ComfyUI may evaluate either.
    graph = _switched_on(T2I)
    _, switch = _outer(graph)
    del switch["on_false"]
    assert expander_prompt.dead_expander(graph) is None
    # The render itself behind a switch that states no choice: it is not sure to run.
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
    _, outer = _outer(graph)
    graph["900"] = {"class_type": "BranchNode", "inputs": {"branch0": list(outer["on_true"])}}
    outer["on_true"] = ["900", 0]
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


def test_an_undecided_node_beside_the_render_path_does_not_turn_the_check_off():
    """Only what stands between the expander and a render matters. A Branch node or a
    switch that states no choice elsewhere in the graph leaves the refusal standing."""
    graph = _switched_on(T2I)
    extract_id, _ = _one(graph, "RegexReplace")
    graph["900"] = {"class_type": "BranchNode", "inputs": {"branch0": [extract_id, 0]}}
    graph["901"] = {"class_type": "ComfySoftSwitchNode",
                    "inputs": {"switch": ["902", 0], "on_true": [extract_id, 0],
                               "on_false": [extract_id, 0]}}
    graph["902"] = {"class_type": "SomeBooleanMaker", "inputs": {}}
    graph["903"] = {"class_type": "PreviewAny", "inputs": {"source": ["901", 0]}}
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
@pytest.mark.parametrize("tap", ["extract", "thinking", "replace"])
def test_an_expander_that_runs_only_for_a_preview_refuses_nothing(name, tap):
    """The switch is false and the task prompt box is empty, and one more Preview
    as Text hangs on the chain. ComfyUI runs the expander for that preview alone; the
    picture takes the request as written, so there is nothing to refuse."""
    graph = _emptied(name)
    extract_id, _ = _one(graph, "RegexReplace")
    expand_id, _ = _one(graph, "TextGenerateV2")
    splice_id, _ = _one(graph, "StringReplace")
    source = {"extract": [extract_id, 0], "thinking": [expand_id, 1], "replace": [splice_id, 0]}[tap]
    graph["900"] = {"class_type": "PreviewAny", "inputs": {"source": source}}
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None
    # The same graph with the switch on feeds the render from the expander, and refuses.
    _, switch = _outer(graph)
    switch["switch"] = True
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_comfyuis_layout_with_the_preview_on_the_text_path_still_refuses(name):
    """ComfyUI's templates wire switch, then Preview as Text, then the conditioning
    prompt. Preview as Text hands its string on, so the render still reads the chain."""
    graph = _emptied(name)
    switch_id, switch = _outer(graph)
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
    # The Fleet sampler renders and cannot call it: it receives no prompt graph. It is
    # a root of the walk all the same, and the UNET loader's call answers for it.
    fleet = pack.NODE_CLASS_MAPPINGS["DGXMonarchFleetKSampler"]
    assert "prompt" not in inspect.signature(fleet.fleet).parameters
    assert set(expander_prompt.ROOT_CLASSES) == {*expander_prompt.RENDER_CLASSES,
                                                 "DGXMonarchFleetKSampler"}


@pytest.mark.parametrize("name", [T2I, EDIT])
def test_a_fleet_render_that_reads_the_expander_refuses_at_the_loader(monkeypatch, name):
    """Fleet takes a per-job positive from a conditioning node. With the expander on
    over an empty task prompt, each job would render from what the expander made of
    nothing, so the loader's call refuses the graph."""
    graph = _switched_on(name)
    sampler_id, sampler = _one(graph, "DGXMonarchKSampler")
    cond_id, _ = _one(graph, "MiniMaxH3ImageConditioning")
    assert sampler["positive"] == [cond_id, 0]
    graph[sampler_id]["class_type"] = "DGXMonarchFleetKSampler"
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, _name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    reached: list = []
    monkeypatch.setattr(loaders, "ensure_live", _explode(reached, "ensure_live"))
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        _explode(reached, "the footprint card"))
    monkeypatch.setattr(render_validation, "preflight_graph_batch_divides_dp",
                        _explode(reached, "the batch preflight"))
    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology",
                        _explode(reached, "the policy card"))
    with pytest.raises(UnsupportedModelError) as raised:
        loaders.DGXMonarchUNETLoader().load(_model().mesh, DIT, prompt=graph)
    assert _empty_sentence(graph) in str(raised.value) and reached == []
    # The same Fleet graph with the switch false sends the request, and refuses nothing.
    _, outer = _outer(graph)
    outer["switch"] = False
    assert expander_prompt.dead_expander(graph) is None


def _switch_graph(choice: object) -> dict:
    """An output that reads a switch whose branches are two text nodes."""
    return {"1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "a"}},
            "2": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "b"}},
            "3": {"class_type": "SomeBooleanMaker", "inputs": {}},
            "4": {"class_type": "ComfySwitchNode",
                  "inputs": {"switch": choice, "on_true": ["1", 0], "on_false": ["2", 0]}},
            "5": {"class_type": "SaveText", "inputs": {"text": ["4", 0]}}}


def test_the_sure_walk_follows_only_what_a_switch_is_certain_to_run():
    # A stated choice reads the same either way: the unselected branch is dropped.
    for choice, kept in ((True, "1"), (False, "2")):
        expected = frozenset({kept, "4", "5", "3"})
        assert graph_live.live_nodes(_switch_graph(choice)) == expected
        assert graph_live.live_nodes(_switch_graph(choice), sure=True) == expected
    # A wired choice: the price keeps both branches, and the sure walk keeps neither,
    # but it keeps what feeds the `switch` input, which ComfyUI must run to choose.
    wired = _switch_graph(["3", 0])
    assert graph_live.live_nodes(wired) == frozenset({"1", "2", "3", "4", "5"})
    assert graph_live.live_nodes(wired, sure=True) == frozenset({"3", "4", "5"})
    # A Branch node's inputs are not followed by the sure walk.
    branch = _switch_graph(True)
    branch["4"] = {"class_type": "BranchNode", "inputs": {"branch0": ["1", 0], "branch1": ["2", 0]}}
    assert graph_live.live_nodes(branch) == frozenset({"1", "2", "3", "4", "5"})
    assert graph_live.live_nodes(branch, sure=True) == frozenset({"3", "4", "5"})
    # Named roots start the same walk.
    assert graph_live.live_nodes(wired, roots=["4"], sure=True) == frozenset({"3", "4"})


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
    def broken(*_args, **_kwargs):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(graph_live, "live_nodes", broken)
    with pytest.raises(RuntimeError):
        expander_prompt.dead_expander(_switched_on(T2I))
    assert expander_prompt.refuse_dead_expander(_switched_on(T2I)) is None


def test_no_other_committed_template_meets_the_refusal_as_shipped_or_switched_on():
    """Every committed graph as shipped, and again with each typed If/Else switch on.
    Only the testing graph, whose task prompt box ships empty, then refuses."""
    assert len(TEMPLATES) > 100
    refused = []
    for name in TEMPLATES:
        graph = _graph(name)
        assert expander_prompt.dead_expander(graph) is None, name
        switches = [node for node in graph.values() if node["class_type"] == "ComfySwitchNode"
                    and isinstance(node["inputs"].get("switch"), bool)]
        for node in switches:
            node["inputs"]["switch"] = True
        if switches and expander_prompt.dead_expander(graph) is not None:
            refused.append(name)
    assert refused == [GRID]


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
