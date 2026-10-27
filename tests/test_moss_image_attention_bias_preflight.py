"""Driver preflight for the MOSS-Image text attention bias (#135).

ComfyUI 9fe7ef764 (in v0.43.0) added image_strength to the Text Encode
MOSS-Image Edit node: a strength other than 1.0 attaches
``context_attention_bias`` to the conditioning, the log of the strength on
each image's tokens and 0 on the others. Stock's forward adds it to the logits
of those text keys, and the sharded attention kernel cannot, so the sharded
forward refuses it on every rank. The driver reads the bias from both prompts
and refuses a MOSS-Image render that carries an effective one under sequence
parallelism before any Worker call, with the same text. Reading both prompts
also covers uly2+cfg2, where a bias on one prompt alone would make one cfg
group refuse in its forward while the other waits at the cfg gather. single,
cfg2 and dp2 bind no forward and apply the bias in stock's, so they pass.

CPU-only: folder_paths is a fake and sniff_checkpoint is monkeypatched, so no
ComfyUI module is imported.
"""
from __future__ import annotations

import ast
import math
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.adapters import detect as detect_mod
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.moss_image_seams import (
    CONTEXT_ATTENTION_BIAS_REFUSAL,
    context_attention_bias_is_effective,
    refuse_context_attention_bias,
)
from dgx_monarch.nodes.common import conditioning_for_wire
from dgx_monarch.nodes.model_patch_preflight import (
    carries_context_attention_bias,
    preflight_moss_image_attention_bias,
)
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.topology import topology_from_preset
from preflight_fails_open_helpers import (
    assert_missing_checkpoint_file_fails_open,
    assert_sniff_checkpoint_raising_fails_open,
    install_fake_folder_paths,
)

REPO = Path(__file__).resolve().parents[1]
PATH = "/models/diffusion_models/moss_image_sft_bf16.safetensors"

# Every preset that binds the sharded forward (a Ulysses or ring degree above
# one), and every preset that runs stock's forward. ring2+fsdp never resolves:
# Topology.validate refuses ring with FSDP.
SHARDED = {preset: topology_from_preset(preset, world) for preset, world in (
    ("uly2", 2), ("ring2", 2), ("uly2+fsdp", 2), ("uly2+cfg2", 4), ("uly4", 4), ("ring4", 4))}
STOCK_FORWARD = {preset: topology_from_preset(preset, world) for preset, world in (
    ("single", 1), ("cfg2", 2), ("dp2", 2), ("cfg2+fsdp", 2), ("dp2+fsdp", 2))}


def _model():
    return types.SimpleNamespace(unet_name="moss_image_sft_bf16.safetensors")


def _sniff(monkeypatch, family="moss_image"):
    calls: list = []
    install_fake_folder_paths(monkeypatch, PATH)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint",
                        lambda path: calls.append(path) or (family, "bf16"))
    return calls


def _strength_bias(strength=0.5, text=6, image=(1, 4)):
    """The edit node's bias: log(strength) on one image's tokens, 0 on the others."""
    bias = [0.0] * text
    bias[image[0]:image[1]] = [math.log(strength)] * (image[1] - image[0])
    return torch.tensor([bias])


def _prompt(extras=None):
    """One prompt as the sampler packs it for the wire: [[tensor, extras]]."""
    return conditioning_for_wire([[torch.zeros(1, 6, 8), dict(extras or {})]])


def _biased(bias):
    return _prompt({"context_attention_bias": bias})


def _request(positive=None, negative=None):
    """A KSampler request at the preflight point, with both prompts packed."""
    return {"kind": "ksampler", "positive": positive if positive is not None else _prompt(),
            "negative": negative if negative is not None else _prompt()}


def _assert_refusal(caught) -> None:
    text = str(caught.value)
    assert text == CONTEXT_ATTENTION_BIAS_REFUSAL
    assert parse_leading_refusal_tag(text).refusal_class is RefusalClass.PHYSICS
    assert text.endswith("See docs/TROUBLESHOOTING.md #135.")


# value, whether stock's forward would change a logit for it
VALUES = {
    "none": (None, False),
    "zeros": (torch.zeros(1, 6), False),
    "bf16-batch-2-zeros": (torch.zeros(2, 6, dtype=torch.bfloat16), False),
    "negative-zero": (torch.full((1, 3), -0.0), False),
    "strength-0.5": (_strength_bias(0.5), True),
    "strength-2": (_strength_bias(2.0), True),
    "nan": (torch.tensor([[0.0, float("nan")]]), True),
    "bool-all-true": (torch.ones(1, 3, dtype=torch.bool), True),
    "integer-zeros": (torch.zeros(1, 3, dtype=torch.int64), True),
    "not-a-tensor": ([[0.0, 0.0]], True),
}
EFFECTIVE = sorted(name for name, (_value, effective) in VALUES.items() if effective)
INERT = sorted(name for name, (_value, effective) in VALUES.items() if not effective)


@pytest.mark.parametrize("name", sorted(VALUES))
def test_the_driver_and_the_worker_decide_on_one_predicate(name):
    value, effective = VALUES[name]
    assert context_attention_bias_is_effective(value) is effective
    if effective:
        with pytest.raises(UnsupportedModelError):
            refuse_context_attention_bias(value)
    else:
        refuse_context_attention_bias(value)
    assert carries_context_attention_bias(_request(positive=_biased(value))) is effective


@pytest.mark.parametrize("preset", sorted(SHARDED))
def test_a_bias_refuses_class_p_under_every_sequence_parallel_preset(monkeypatch, preset):
    calls = _sniff(monkeypatch)
    with pytest.raises(UnsupportedModelError) as caught:
        preflight_moss_image_attention_bias(
            _model(), _request(positive=_biased(_strength_bias())), SHARDED[preset])
    _assert_refusal(caught)
    assert calls == [PATH]


@pytest.mark.parametrize("preset", ["uly2", "ring2", "uly2+cfg2"])
def test_a_bias_on_the_negative_alone_refuses(monkeypatch, preset):
    """The edit node's guide asks for it on the negative too; on uly2+cfg2 that prompt
    reaches only the uncond group, so the driver is the only place both groups refuse."""
    _sniff(monkeypatch)
    with pytest.raises(UnsupportedModelError) as caught:
        preflight_moss_image_attention_bias(
            _model(), _request(negative=_biased(_strength_bias(3.0))), SHARDED[preset])
    _assert_refusal(caught)


@pytest.mark.parametrize("preset", sorted(STOCK_FORWARD))
def test_stock_forward_presets_pass_without_reading_a_header(monkeypatch, preset):
    calls = _sniff(monkeypatch)
    request = _request(positive=_biased(_strength_bias()), negative=_biased(_strength_bias()))
    preflight_moss_image_attention_bias(_model(), request, STOCK_FORWARD[preset])
    assert calls == []


@pytest.mark.parametrize("name", EFFECTIVE)
def test_every_effective_value_refuses_on_uly2(monkeypatch, name):
    """A boolean bias refuses too: stock pads it with False over the image keys."""
    _sniff(monkeypatch)
    with pytest.raises(UnsupportedModelError) as caught:
        preflight_moss_image_attention_bias(
            _model(), _request(positive=_biased(VALUES[name][0])), SHARDED["uly2"])
    _assert_refusal(caught)


@pytest.mark.parametrize("name", INERT)
def test_an_inert_bias_passes_without_reading_a_header(monkeypatch, name):
    """An all-zero bias adds nothing in stock, and the sharded forward runs without it."""
    calls = _sniff(monkeypatch)
    request = _request(positive=_biased(VALUES[name][0]), negative=_biased(VALUES[name][0]))
    preflight_moss_image_attention_bias(_model(), request, SHARDED["uly2"])
    assert calls == []


def test_a_render_without_the_edit_node_never_reads_a_header(monkeypatch):
    calls = _sniff(monkeypatch)
    preflight_moss_image_attention_bias(_model(), _request(), SHARDED["uly2"])
    preflight_moss_image_attention_bias(_model(), {}, SHARDED["uly2"])
    assert calls == []


@pytest.mark.parametrize("family", ["krea2", "flux3", "qwen_image"])
def test_other_families_are_left_to_their_own_guards(monkeypatch, family):
    calls = _sniff(monkeypatch, family)
    preflight_moss_image_attention_bias(
        _model(), _request(positive=_biased(_strength_bias())), SHARDED["uly2"])
    assert calls == [PATH]


def test_sniff_checkpoint_raising_fails_open(monkeypatch):
    assert_sniff_checkpoint_raising_fails_open(
        monkeypatch, preflight=preflight_moss_image_attention_bias, model=_model(),
        refusing_args=(_request(positive=_biased(_strength_bias())), SHARDED["uly2"]),
        default_path=PATH)


def test_missing_checkpoint_file_fails_open(monkeypatch):
    assert_missing_checkpoint_file_fails_open(
        monkeypatch, preflight=preflight_moss_image_attention_bias, model=_model(),
        refusing_args=(_request(positive=_biased(_strength_bias())), SHARDED["uly2"]),
        family="moss_image")


_BIAS = _strength_bias()
MALFORMED = {
    "request-none": None,
    "request-list": [("positive", [[torch.zeros(1), {"context_attention_bias": _BIAS}]])],
    "prompts-none": {"positive": None, "negative": None},
    "prompt-string": {"positive": "context_attention_bias"},
    "prompt-dict": {"positive": {"context_attention_bias": _BIAS}},
    "entries-not-lists": {"positive": [None, "entry", 3, _BIAS]},
    "entry-without-extras": {"positive": [[torch.zeros(1)]]},
    "extras-none": {"positive": [[torch.zeros(1), None]]},
    "extras-string": {"positive": [[torch.zeros(1), "context_attention_bias"]]},
    "extras-pairs": {"positive": [[torch.zeros(1), [("context_attention_bias", _BIAS)]]]},
    "guider-string": {"kind": "custom", "guider": "cfg"},
    "guider-prompts-none": {"kind": "custom", "guider": {"kind": "dual_model", "positive": None,
                                                         "negative": None}},
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_conditioning_passes_and_never_raises(monkeypatch, name):
    calls = _sniff(monkeypatch)
    request = MALFORMED[name]
    if isinstance(request, dict):
        assert carries_context_attention_bias(request) is False
    preflight_moss_image_attention_bias(_model(), request, SHARDED["uly2"])
    assert calls == []


class _UnreadableBias(torch.Tensor):
    def any(self, *args, **kwargs):
        raise RuntimeError("this bias cannot be read")


def test_a_bias_the_predicate_cannot_read_passes_on_the_driver(monkeypatch):
    """The driver fails open where it cannot read its facts; the Worker's forward meets it."""
    calls = _sniff(monkeypatch)
    unreadable = torch.zeros(1, 3).as_subclass(_UnreadableBias)
    with pytest.raises(RuntimeError, match="cannot be read"):
        refuse_context_attention_bias(unreadable)
    request = {"positive": [[torch.zeros(1), {"context_attention_bias": unreadable}]]}
    preflight_moss_image_attention_bias(_model(), request, SHARDED["uly2"])
    assert calls == []


GUIDERS = {
    "basic": {"kind": "basic", "positive": _biased(_BIAS)},
    "cfg-positive": {"kind": "cfg", "positive": _biased(_BIAS), "negative": _prompt(), "cfg": 3.0},
    "cfg-negative": {"kind": "cfg", "positive": _prompt(), "negative": _biased(_BIAS), "cfg": 3.0},
    "dual-model": {"kind": "dual_model", "positive": _biased(_BIAS), "negative": None, "cfg": 3.0},
}


@pytest.mark.parametrize("name", sorted(GUIDERS))
def test_samplercustom_guider_prompts_count(monkeypatch, name):
    """SamplerCustom carries its prompts in the guider spec, not on the request."""
    _sniff(monkeypatch)
    request = {"kind": "custom", "guider": GUIDERS[name], "uncond_model": None}
    with pytest.raises(UnsupportedModelError) as caught:
        preflight_moss_image_attention_bias(_model(), request, SHARDED["uly2"])
    _assert_refusal(caught)


@pytest.mark.parametrize("negative", [None, [[None, {}]]], ids=["dual-model-none", "cfg-placeholder"])
def test_a_guider_without_a_bias_or_a_negative_passes(monkeypatch, negative):
    """The dual-model guider sends None, or [[None, {}]] when it falls back to CFG."""
    calls = _sniff(monkeypatch)
    request = {"kind": "custom", "guider": {"kind": "cfg", "positive": _prompt(), "negative": negative}}
    preflight_moss_image_attention_bias(_model(), request, SHARDED["uly2"])
    assert calls == []


def _call_order(path: Path, function: str) -> list[str]:
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    return [ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)]


def test_render_submit_runs_the_bias_preflight_once_right_after_the_reference_preflight():
    calls = _call_order(REPO / "src/dgx_monarch/nodes/render_submit.py", "_submit_render_guarded")
    bias = "model_patch_preflight.preflight_moss_image_attention_bias"
    assert calls.count(bias) == 1
    assert calls[calls.index(bias) - 1] == "model_patch_preflight.preflight_moss_image_reference_images"
    assert calls.index(bias) < calls.index("mesh_setup.ensure_request_setup")


@pytest.mark.parametrize("request_kind", ["ksampler-negative-uly2+cfg2", "custom-uly2"])
def test_preflight_runs_before_worker_dispatch(monkeypatch, request_kind):
    """_submit_render_guarded refuses from this preflight before any mesh_setup call."""
    from dgx_monarch.nodes import common, render_submit
    from dgx_monarch.nodes.submit_guard import SubmitRenderGuard

    _sniff(monkeypatch)

    def _fail(*_a, **_k):
        pytest.fail("worker dispatch must not run when the driver preflight refuses")

    for name in ("prepare_sample", "submit_sample", "ensure_request_setup"):
        monkeypatch.setattr(render_submit.mesh_setup, name, _fail, raising=False)

    preset = "uly2+cfg2" if request_kind.endswith("uly2+cfg2") else "uly2"

    class _Handle:
        world = SHARDED[preset].world

        def request_setup(self, *_a, **_k):
            _fail()

    class _MeshSpec:
        handle = _Handle()
        topology_preset = preset
        attention = "auto"
        world = SHARDED[preset].world

    model = types.SimpleNamespace(mesh=_MeshSpec(), unet_name="moss_image_sft_bf16.safetensors",
                                  options={})
    monkeypatch.setattr(common, "_bind_packed_render_model", lambda model, latent, cfg: (model, None))
    monkeypatch.setattr(render_submit, "model_for_request", lambda model, request: model)
    monkeypatch.setattr(common, "ensure_live", lambda handle: handle)
    monkeypatch.setattr(common, "_resolve_topology_for_latent",
                        lambda model, latent, cfg, world: (SHARDED[preset], False, f"preset: {preset}"))
    if request_kind.startswith("custom"):
        request = {"kind": "custom", "guider": GUIDERS["cfg-positive"], "uncond_model": None,
                   "_dgxm_krea2_ref_preflight": None}
    else:
        request = _request(negative=_biased(_BIAS))
    with pytest.raises(UnsupportedModelError) as caught:
        render_submit._submit_render_guarded(
            model, request, {"samples": None}, cfg_value=3.0, steps_hint=50, seq=0, depth=1,
            progress_on_step=None, handoff=None, guard=SubmitRenderGuard())
    _assert_refusal(caught)
