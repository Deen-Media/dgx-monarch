"""Shape pins for the WanDancer global-to-local bridge workflow fixture.

CPU only: valid JSON with the API-format envelope, only allowlisted stock and
Monarch node classes, no `reference_latent` input, and two sampler stages
joined by stock nodes. It never submits the graph or touches hardware.
"""
import os

from workflow_pin_helpers import assert_uses_only_known_node_classes, load_workflow_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.join(
    HERE, "fixtures", "workflows", "wandancer_global_local_uly2_api.json"
)

# Every class_type the fixture may reference: dgx-monarch mesh and sampler
# nodes plus the stock ComfyUI nodes in docs/MODELS.md's WanDancer two-pass
# contract. This is the fixture's allowlist, not a node registry: add a node
# here only when the fixture uses it.
_KNOWN_CLASS_TYPES = frozenset({
    "DGXMonarchInit",
    "DGXMonarchUNETLoader",
    "DGXMonarchKSampler",
    "CLIPLoader",
    "CLIPTextEncode",
    "VAELoader",
    "VAEDecode",
    "LoadAudio",
    "WanDancerEncodeAudio",
    "WanDancerVideo",
    "WanDancerPadKeyframesList",
    "SaveImage",
    # Required image conditioning (2026-07-28): WanDancer produces
    # non-finite latents without start_image + clip_vision_output.
    "CLIPVisionLoader",
    "LoadImage",
    "CLIPVisionEncode",
})


def _load_prompt() -> dict:
    return load_workflow_prompt(WORKFLOW)


def test_wandancer_bridge_workflow_is_valid_json_with_meta_envelope():
    _load_prompt()


def test_wandancer_bridge_workflow_uses_only_known_node_classes():
    assert_uses_only_known_node_classes(_load_prompt(), _KNOWN_CLASS_TYPES)


def test_wandancer_bridge_workflow_never_wires_reference_latent():
    prompt = _load_prompt()
    for node_id, node in prompt.items():
        assert "reference_latent" not in node["inputs"], (
            f"node {node_id!r} wires reference_latent, which the WanDancer "
            "adapter typed-rejects (src/dgx_monarch/adapters/wan_variants.py)"
        )


def test_wandancer_bridge_workflow_has_two_sampler_stages_bridged_by_stock_nodes():
    prompt = _load_prompt()
    samplers = [
        node_id for node_id, node in prompt.items()
        if node["class_type"] == "DGXMonarchKSampler"
    ]
    assert len(samplers) == 2, "expected one global-stage and one local-stage sampler"

    decodes = [
        node_id for node_id, node in prompt.items()
        if node["class_type"] == "VAEDecode"
    ]
    assert len(decodes) == 2, "expected one decode per sampler stage"

    pad_keyframes = [
        node_id for node_id, node in prompt.items()
        if node["class_type"] == "WanDancerPadKeyframesList"
    ]
    assert len(pad_keyframes) == 1, "expected exactly one keyframe-bridge node"
