"""The loader's batch preflight counts only the samplers that take its model.

``preflight_graph_batch_divides_dp`` refuses a literal latent batch that the
loader's explicit DP degree cannot split, before any weights load. A graph can
hold other branches: a Fleet KSampler that runs independent world-one jobs, or
a sampler fed by another loader on another mesh. Their batches never split
across this loader's DP ranks, so they must not refuse this load. An
association the walk cannot prove makes no claim here; the render submit still
checks the real tensor against the topology it resolved.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch.nodes import render_validation


def _mesh(preset="dp2", world=2):
    return types.SimpleNamespace(
        topology_preset=preset,
        handle=types.SimpleNamespace(world=world, defunct=False))


def _latent(batch):
    return {"class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": batch}}


def _loader():
    return {"class_type": "DGXMonarchUNETLoader",
            "inputs": {"mesh": ["1", 0], "unet_name": "model.safetensors"}}


def _sampler(model_link, latent_link, class_type="DGXMonarchKSampler"):
    return {"class_type": class_type,
            "inputs": {"model": model_link, "latent_image": latent_link}}


def _refuses(prompt, loader_id="2", mesh=None):
    try:
        render_validation.preflight_graph_batch_divides_dp(
            mesh or _mesh(), prompt, loader_id)
    except ValueError as exc:
        assert "is not divisible by 2" in str(exc)
        return True
    return False


def test_a_dp2_mesh_feeding_fleet_with_batch_one_is_admitted():
    """Fleet dispatches one world-one job per GPU: no batch splits over DP."""
    prompt = {"2": _loader(), "8": _latent(1),
              "9": _sampler(["2", 0], ["8", 0], "DGXMonarchFleetKSampler")}
    assert not _refuses(prompt)


def test_fleet_beside_an_ordinary_sampler_still_prices_the_ordinary_one():
    prompt = {"2": _loader(), "8": _latent(1), "18": _latent(1),
              "9": _sampler(["2", 0], ["8", 0], "DGXMonarchFleetKSampler"),
              "10": _sampler(["2", 0], ["18", 0])}
    assert _refuses(prompt)


def _mixed():
    """Loader 2 feeds a batch-two sampler, loader 3 a batch-one sampler."""
    return {"2": _loader(), "3": _loader(), "8": _latent(2), "18": _latent(1),
            "9": _sampler(["2", 0], ["8", 0]),
            "10": _sampler(["3", 0], ["18", 0])}


def test_another_loaders_batch_one_branch_does_not_block_this_dp2_branch():
    assert not _refuses(_mixed(), loader_id="2")


def test_the_other_loader_still_answers_for_its_own_branch():
    assert _refuses(_mixed(), loader_id="3")


def test_a_lora_and_skip_layer_chain_still_reaches_its_loader():
    prompt = {"2": _loader(), "8": _latent(1),
              "5": {"class_type": "DGXMonarchLoraLoader",
                    "inputs": {"model": ["2", 0], "lora_name": "a.safetensors",
                               "strength_model": 1.0}},
              "6": {"class_type": "DGXMonarchSkipLayerGuidance",
                    "inputs": {"model": ["5", 0]}},
              "9": _sampler(["6", 0], ["8", 0])}
    assert _refuses(prompt)


def test_sampler_custom_reaches_its_loader_through_the_guider():
    prompt = {"2": _loader(), "8": _latent(1),
              "6": {"class_type": "DGXMonarchCFGGuider",
                    "inputs": {"model": ["2", 0], "cfg": 4.0}},
              "9": {"class_type": "DGXMonarchSamplerCustom",
                    "inputs": {"guider": ["6", 0], "latent_image": ["8", 0]}}}
    assert _refuses(prompt)


def test_the_uncond_loader_answers_for_a_dual_model_guider_it_feeds():
    prompt = {"2": _loader(), "3": dict(_loader(), class_type="DGXMonarchUncondUNETLoader"),
              "8": _latent(1),
              "6": {"class_type": "DGXMonarchDualModelGuider",
                    "inputs": {"model": ["2", 0], "model_negative": ["3", 0]}},
              "9": {"class_type": "DGXMonarchSamplerCustom",
                    "inputs": {"guider": ["6", 0], "latent_image": ["8", 0]}}}
    assert _refuses(prompt, loader_id="3")
    assert _refuses(prompt, loader_id="2")


@pytest.mark.parametrize("prompt,loader_id", [
    # No loader id: nothing ties a sampler to this load.
    ({"2": _loader(), "8": _latent(1), "9": _sampler(["2", 0], ["8", 0])}, None),
    # A node the walk does not know sits between the loader and the sampler.
    ({"2": _loader(), "8": _latent(1),
      "5": {"class_type": "SomeCustomModelPatch", "inputs": {"model": ["2", 0]}},
      "9": _sampler(["5", 0], ["8", 0])}, "2"),
    # A lazy switch picks the model: either branch may run.
    ({"2": _loader(), "3": _loader(), "8": _latent(1),
      "5": {"class_type": "ComfySwitchNode",
            "inputs": {"switch": ["7", 0], "on_true": ["2", 0], "on_false": ["3", 0]}},
      "9": _sampler(["5", 0], ["8", 0])}, "2"),
    # A broken link.
    ({"2": _loader(), "8": _latent(1), "9": _sampler(["404", 0], ["8", 0])}, "2"),
    # A model input that is not a link at all.
    ({"2": _loader(), "8": _latent(1), "9": _sampler(3, ["8", 0])}, "2"),
])
def test_an_unproven_association_makes_no_claim_at_the_loader(prompt, loader_id):
    assert not _refuses(prompt, loader_id=loader_id)


def test_a_cycle_in_a_malformed_graph_makes_no_claim():
    prompt = {"2": _loader(), "8": _latent(1),
              "5": {"class_type": "DGXMonarchLoraLoader", "inputs": {"model": ["6", 0]}},
              "6": {"class_type": "DGXMonarchLoraLoader", "inputs": {"model": ["5", 0]}},
              "9": _sampler(["5", 0], ["8", 0])}
    assert not _refuses(prompt)


def test_the_loader_node_passes_its_own_id(monkeypatch):
    """The node hands ComfyUI's UNIQUE_ID to the preflight, so the walk has
    a loader to associate with."""
    from dgx_monarch.nodes import loader_preflight, loaders

    seen: list[tuple] = []

    def record(mesh, prompt, loader_id=None):
        seen.append((prompt, loader_id))
        raise RuntimeError("stop after the batch preflight")

    monkeypatch.setattr(loader_preflight, "preflight_upstream_gated_artifact",
                        lambda *_a: None)
    monkeypatch.setattr(render_validation, "preflight_graph_batch_divides_dp", record)
    mesh = types.SimpleNamespace(topology_preset="auto", worker_args={})
    with pytest.raises(RuntimeError, match="stop after the batch preflight"):
        loaders.DGXMonarchUNETLoader().load(
            mesh, "model.safetensors", unique_id="2", prompt={"2": _loader()})
    assert seen == [({"2": _loader()}, "2")]


def test_the_walk_knows_every_registered_node_that_hands_on_a_model():
    """Census: a new dgx-monarch node that returns a model or a guider must
    join the walk's table, or its graphs silently lose the early answer."""
    from dgx_monarch.constants import GUIDER_TYPE, MODEL_TYPE
    from dgx_monarch.nodes import NODE_CLASS_MAPPINGS

    loaders = {"DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader"}
    handing_on = {name for name, cls in NODE_CLASS_MAPPINGS.items()
                  if name not in loaders
                  and set(getattr(cls, "RETURN_TYPES", ())) & {MODEL_TYPE, GUIDER_TYPE}}
    assert handing_on == set(render_validation._MODEL_LINK_KEYS)
    for name in handing_on:
        try:
            spec = NODE_CLASS_MAPPINGS[name].INPUT_TYPES()
        except ImportError:
            continue  # its schema needs ComfyUI; the name check above still holds
        carried = {key for section in spec.values() if isinstance(section, dict)
                   for key, value in section.items()
                   if isinstance(value, tuple) and value and value[0] in (MODEL_TYPE, GUIDER_TYPE)}
        assert carried == set(render_validation._MODEL_LINK_KEYS[name]), name


def test_every_shipped_sampler_reaches_its_loader():
    """The shipped templates keep the early answer: each sampler that reads a
    latent, Fleet aside, walks back to a dgx-monarch loader in its graph."""
    import json
    from pathlib import Path

    from benchmark.sweep import convert

    root = Path(__file__).resolve().parents[1]
    names = convert.repo_widget_names(root)
    checked, lost = 0, []
    for path in sorted((root / "example_workflows").glob("*.json")):
        graph = convert.convert(json.loads(path.read_text()), names)
        loader_ids = [key for key, node in graph.items() if node.get("class_type") in (
            "DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader")]
        for key, node in graph.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            if (not class_type.startswith("DGXMonarch") or class_type == "DGXMonarchFleetKSampler"
                    or "latent_image" not in inputs):
                continue
            checked += 1
            if not any(render_validation._takes_loader(graph, inputs, str(loader))
                       for loader in loader_ids):
                lost.append(f"{path.name}:{key}")
    assert checked > 50
    assert lost == []
