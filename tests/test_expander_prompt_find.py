"""The empty-task-prompt refusal reads Replace Text's ``find`` as ComfyUI runs it.

Replace Text hands on its text with every ``find`` replaced by the request, so
over an empty task prompt the request still reaches Generate Text when the find
is empty too (``"".replace("", request)`` is the request), and the check cannot
say what comes through a find the graph does not state. Neither refuses. A find
that holds the marker over an empty task prompt still refuses. The shipped
MiniMax-H3-Image graphs, converted as the sweep converts them, run through the
refusal the loader, sampler and Identity Gate raise.
"""
from __future__ import annotations

import pytest

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import expander_prompt
from test_expander_prompt import FAMILY_GRAPHS, _empty_sentence, _one, _switched_on


def _with_find(name: str, task: str, find: object) -> dict:
    graph = _switched_on(name, task)
    _, splice = _one(graph, "StringReplace")
    splice["find"] = find
    return graph


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
@pytest.mark.parametrize("task, find", [("", ""), (" ", " "), ("\n", "\n")])
def test_an_empty_find_over_an_empty_task_prompt_hands_on_the_request(name, task, find):
    graph = _with_find(name, task, find)
    # What ComfyUI's Replace Text hands Generate Text: the request itself.
    assert task.replace(find, "a red bicycle").strip() == "a red bicycle"
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_a_find_the_graph_does_not_state_refuses_nothing(name):
    # Wired to a node whose text the graph does not state.
    graph = _with_find(name, "", ["find_maker", 0])
    graph["find_maker"] = {"class_type": "SomeTextMaker", "inputs": {}}
    assert expander_prompt.dead_expander(graph) is None
    assert expander_prompt.refuse_dead_expander(graph) is None


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_a_find_from_a_text_box_is_read_as_a_typed_one(name):
    graph = _with_find(name, "", ["find_box", 0])
    graph["find_box"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": ""}}
    assert expander_prompt.dead_expander(graph) is None
    graph["find_box"]["inputs"]["value"] = "{input text}"
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)


@pytest.mark.parametrize("name", FAMILY_GRAPHS)
def test_the_marker_over_an_empty_task_prompt_still_refuses(name):
    graph = _with_find(name, "", "{input text}")
    assert expander_prompt.dead_expander(graph) == _empty_sentence(graph)
    with pytest.raises(UnsupportedModelError):
        expander_prompt.refuse_dead_expander(graph)
