"""Structural pin for the MiniMax H3 t2va workflow fixture. CPU-only: this
asserts the fixture's shape (valid JSON, valid API-format envelope, only known
stock/monarch node classes, a topology H3 can run, one packed AV latent decoded
through two VAEs); it never POSTs the graph or touches hardware.
"""
import os

from workflow_pin_helpers import assert_uses_only_known_node_classes, load_workflow_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.join(
    HERE, "fixtures", "workflows", "minimax_h3_t2va_uly2_api.json"
)

# Topologies H3 typed-refuses: comfy runs each conditioning as its own batch-1
# call, so cfg-parallel has no batched call to split, and the NestedTensor
# latent cannot be data-split (docs/TROUBLESHOOTING.md #48).
_REFUSED_TOPOLOGIES = frozenset({"cfg2", "dp2"})

# Every class_type the fixture is allowed to reference: dgx-monarch mesh/sampler
# nodes plus the stock ComfyUI nodes the MiniMax H3 t2va path needs. Adding a
# node here means adding it to the workflow too; this list is the fixture's
# allowlist, not a general node registry.
_KNOWN_CLASS_TYPES = frozenset({
    "DGXMonarchInit",
    "DGXMonarchUNETLoader",
    "DGXMonarchKSampler",
    "CLIPLoader",
    "CLIPTextEncode",
    "VAELoader",
    "EmptyMiniMaxH3LatentAV",
    "MiniMaxH3ImageToVideo",
    "VAEDecode",
    "VAEDecodeAudio",
    "SaveImage",
    "SaveAudioAdvanced",
})


def _load_prompt() -> dict:
    return load_workflow_prompt(WORKFLOW)


def _nodes_of(prompt: dict, class_type: str) -> list:
    return [node_id for node_id, node in prompt.items() if node["class_type"] == class_type]


def test_minimax_h3_workflow_is_valid_json_with_meta_envelope():
    _load_prompt()


def test_minimax_h3_workflow_uses_only_known_node_classes():
    assert_uses_only_known_node_classes(_load_prompt(), _KNOWN_CLASS_TYPES)


def test_minimax_h3_workflow_has_exactly_one_monarch_sampler():
    prompt = _load_prompt()
    samplers = _nodes_of(prompt, "DGXMonarchKSampler")
    assert len(samplers) == 1, "expected exactly one sampler stage"


def test_minimax_h3_workflow_never_requests_a_refused_topology():
    prompt = _load_prompt()
    init_nodes = _nodes_of(prompt, "DGXMonarchInit")
    assert len(init_nodes) == 1, "expected exactly one Init node"
    topology = prompt[init_nodes[0]]["inputs"]["topology"]
    assert topology not in _REFUSED_TOPOLOGIES, (
        f"Init node requests {topology!r}, which MiniMax H3 typed-refuses "
        "(src/dgx_monarch/adapters/minimax_h3.py)"
    )


def test_minimax_h3_workflow_never_wires_a_batch_above_one():
    prompt = _load_prompt()
    for node_id, node in prompt.items():
        batch_size = node["inputs"].get("batch_size")
        if batch_size is None:
            continue
        assert int(batch_size) == 1, (
            f"node {node_id!r} wires batch_size {batch_size!r}; stock ComfyUI "
            "raises on any MiniMax H3 batch above 1"
        )


def test_minimax_h3_workflow_decodes_one_packed_latent_through_two_vaes():
    prompt = _load_prompt()
    sampler = _nodes_of(prompt, "DGXMonarchKSampler")[0]
    video_decodes = _nodes_of(prompt, "VAEDecode")
    audio_decodes = _nodes_of(prompt, "VAEDecodeAudio")
    assert len(video_decodes) == 1 and len(audio_decodes) == 1, (
        "expected one video decode and one audio decode"
    )

    video_vae = prompt[video_decodes[0]]["inputs"]["vae"]
    audio_vae = prompt[audio_decodes[0]]["inputs"]["vae"]
    assert video_vae != audio_vae, "video and audio streams need their own VAE"
    for decode in (video_decodes[0], audio_decodes[0]):
        assert prompt[decode]["inputs"]["samples"] == [sampler, 0], (
            f"node {decode!r} must decode the sampler's packed AV latent"
        )
