"""Keep every template's literal-latent provider in the pre-load batch census.

The loader reads the empty-latent node's literal batch_size to reject an
indivisible dp2 batch before loading weights. Missing a provider defers the
refusal until sampling, after the checkpoint has loaded.

Every Empty*Latent and NATIVE_LITERAL_PROVIDERS node in shipped or testing-only
templates must be classified in LITERAL_BATCH_LATENT_CLASSES (priced) or
NO_LITERAL_BATCH_LATENT_CLASSES (knowingly unpriced). New providers require an
explicit choice.

The census reads shipped UI graphs (a nodes list keyed by type). Behavior tests
build API prompts (a mapping keyed by class_type), which the runtime check reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dgx_monarch.nodes import render_validation
from dgx_monarch.refusal import parse_leading_refusal_tag

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / "example_workflows"

# The four classes the 2026-09-03 census found missing that state a literal
# batch_size, copied from the ComfyUI node schemas the templates target.
NEWLY_PRICED = (
    "EmptyFlux2LatentImage",
    "EmptyChromaRadianceLatentImage",
    "EmptyHunyuanImageLatent",
    "EmptyHunyuanVideo15Latent",
)

NATIVE_LITERAL_PROVIDERS = (
    "TextEncodeQwenImage21",
    "WanAnimate2ToVideo",
)


def _template_paths() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.json")) + sorted(
        (REPO / "tests" / "fixtures" / "workflows" / "generated").glob("*.json"))


def _latent_classes(path: Path) -> set[str]:
    """Literal-latent node types this template uses, from the UI graph.

    Assert the shape: ``graph.get("nodes") or ()`` over anything else walks
    nothing, so a template saved as an API prompt would pass this census with
    no class read.
    """
    graph = json.loads(path.read_text())
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    assert isinstance(nodes, list), (
        f"{path.name} is not a ComfyUI UI graph: no top-level `nodes` list, so "
        "this census reads no latent class out of it at all. Teach this walk "
        "the shape the file is in rather than leaving it walking nothing.")
    found = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("type")
        if (isinstance(name, str)
                and ((name.startswith("Empty") and "Latent" in name)
                     or name in NATIVE_LITERAL_PROVIDERS)):
            found.add(name)
    return found


def test_a_template_shape_this_walk_cannot_read_fails_the_census(tmp_path):
    """A file the walk reads nothing from must fail, not pass.

    An API prompt keys nodes by id and carries no top-level ``nodes`` list, so
    the loop would run zero times. On 2026-10-06 all 82 templates, shipped and
    testing-only, were UI graphs, and none held a subgraph definition. This walk
    does not read the nodes inside a subgraph definition.
    """
    prompt = tmp_path / "api-format.json"
    prompt.write_text(json.dumps(
        {"8": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1}}}))
    with pytest.raises(AssertionError, match="not a ComfyUI UI graph"):
        _latent_classes(prompt)


def test_the_template_set_is_not_empty():
    """A walk over no files asserts nothing and always passes."""
    paths = _template_paths()
    assert len(paths) > 20
    assert any(path.parent == REPO / "tests" / "fixtures" / "workflows" / "generated"
               for path in paths)


def test_every_template_latent_class_is_censused():
    """No template uses a latent class outside the census, the gap the 2026-09-03 leg hit."""
    uncensused: dict[str, list[str]] = {}
    for path in _template_paths():
        for name in sorted(_latent_classes(path)):
            if name not in render_validation.CENSUSED_LATENT_CLASSES:
                uncensused.setdefault(name, []).append(path.name)
    assert not uncensused, (
        "these empty-latent classes are used by a shipped template and are in "
        "neither LITERAL_BATCH_LATENT_CLASSES nor "
        f"NO_LITERAL_BATCH_LATENT_CLASSES: {uncensused}. A class that is not "
        "censused makes the loader-site batch check a no-op for its templates, "
        "so an indivisible batch costs a full checkpoint load before the "
        "sampler site answers.")


def test_the_named_literal_providers_are_all_censused():
    """The 2026-09-03 additions, the native providers and the packed AV latent stay censused."""
    for name in (*NEWLY_PRICED, *NATIVE_LITERAL_PROVIDERS,
                 "EmptyMiniMaxH3LatentAV"):
        assert name in render_validation.CENSUSED_LATENT_CLASSES


def test_the_two_census_halves_do_not_overlap():
    """A class is priced or knowingly unpriced, never both."""
    assert not (render_validation.LITERAL_BATCH_LATENT_CLASSES
                & render_validation.NO_LITERAL_BATCH_LATENT_CLASSES)


def test_the_packed_av_latent_is_censused_unpriced():
    """Its node states width, height and length and no batch_size.

    Listing it as priced would claim a literal this walk cannot read. Its dp2
    and cfg2 answer is the packed-latent refusal at the submit
    (``validate_render_topology``).
    """
    assert ("EmptyMiniMaxH3LatentAV"
            in render_validation.NO_LITERAL_BATCH_LATENT_CLASSES)
    assert ("EmptyMiniMaxH3LatentAV"
            not in render_validation.LITERAL_BATCH_LATENT_CLASSES)


def _mesh(preset="dp2", world=2):
    class _Handle:
        defunct = False

    handle = _Handle()
    handle.world = world

    class _Mesh:
        topology_preset = preset

    mesh = _Mesh()
    mesh.handle = handle
    return mesh


def _prompt(latent_class, batch):
    """The API prompt shape the loader node hands the check."""
    return {
        "8": {"class_type": latent_class,
              "inputs": {"width": 1024, "height": 1024, "batch_size": batch}},
        "9": {"class_type": "DGXMonarchKSampler",
              "inputs": {"model": ["2", 0], "latent_image": ["8", 0]}},
    }


@pytest.mark.parametrize("latent_class", NEWLY_PRICED)
def test_a_newly_priced_class_refuses_an_indivisible_batch_at_the_loader(latent_class):
    """Unpriced, each of these loads the whole checkpoint before the sampler refuses."""
    with pytest.raises(ValueError) as raised:
        render_validation.preflight_graph_batch_divides_dp(
            _mesh(), _prompt(latent_class, 1))
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class.value == "P"
    assert "latent batch 1 is not divisible by 2" in text


@pytest.mark.parametrize("latent_class", NEWLY_PRICED)
def test_a_newly_priced_class_admits_a_divisible_batch(latent_class):
    render_validation.preflight_graph_batch_divides_dp(
        _mesh(), _prompt(latent_class, 2))


def test_qwen21_fixed_batch_one_refuses_at_the_loader_only_from_its_latent_output():
    prompt = _prompt("TextEncodeQwenImage21", 1)
    prompt["8"]["inputs"] = {"resolution": 1024}
    prompt["9"]["inputs"]["latent_image"] = ["8", 2]
    with pytest.raises(ValueError, match="latent batch 1 is not divisible by 2"):
        render_validation.preflight_graph_batch_divides_dp(_mesh(), prompt)

    # Output zero is positive conditioning, not a sampling latent.
    prompt["9"]["inputs"]["latent_image"] = ["8", 0]
    render_validation.preflight_graph_batch_divides_dp(_mesh(), prompt)

    # A batch_size input on this node makes no claim: the guard does not assume
    # batch one for a schema it has not been taught.
    prompt["9"]["inputs"]["latent_image"] = ["8", 2]
    prompt["8"]["inputs"]["batch_size"] = ["other", 0]
    render_validation.preflight_graph_batch_divides_dp(_mesh(), prompt)


def test_wan_animate2_reads_only_a_literal_batch_size_from_its_latent_output():
    prompt = _prompt("WanAnimate2ToVideo", 1)
    prompt["9"]["inputs"]["latent_image"] = ["8", 2]
    with pytest.raises(ValueError, match="latent batch 1 is not divisible by 2"):
        render_validation.preflight_graph_batch_divides_dp(_mesh(), prompt)

    prompt["8"]["inputs"]["batch_size"] = 2
    render_validation.preflight_graph_batch_divides_dp(_mesh(), prompt)

    prompt["8"]["inputs"]["batch_size"] = ["other", 0]
    render_validation.preflight_graph_batch_divides_dp(_mesh(), prompt)

    # Output zero is positive conditioning, never the Animate2 latent.
    prompt["8"]["inputs"]["batch_size"] = 1
    prompt["9"]["inputs"]["latent_image"] = ["8", 0]
    render_validation.preflight_graph_batch_divides_dp(_mesh(), prompt)


def test_the_unpriced_class_still_makes_no_claim_at_the_loader():
    """Censused is not priced: the check reads no batch from this class, so it makes no claim."""
    render_validation.preflight_graph_batch_divides_dp(
        _mesh(), {"8": {"class_type": "EmptyMiniMaxH3LatentAV",
                        "inputs": {"width": 1344, "height": 768, "length": 124}},
                  "9": {"class_type": "DGXMonarchKSampler",
                        "inputs": {"latent_image": ["8", 0]}}})
