"""What a template Note says about the identity gate and FSDP matches the code.

The gate: a KSampler render of a risky combination runs the first-use gate,
which compares renders on the graph's own mesh and topology, never a stock
single-GPU load (nodes/gate_ceremony.py, nodes/gate_cross_mode.py). A custom
sampler render runs no automatic gate, and without a PASS it dispatches on
stock residency (auto_gate.auto_gate_context, gate_identity.authorize_normal_render).
A dual-model render runs no gate either and always dispatches on stock
residency (auto_gate.maybe_auto_gate). Each graph's Note must say the sentence
for its own sampler.

FSDP: the launch contract shards bf16, fp16, fp8 and int8 files and refuses
block-scaled ones (adapters/fsdp.py), a LoRA there needs lora_low_rss and a
file that is not comfy-kitchen quantized (adapters/fsdp_lora_admission.py),
and Anima's llm_adapter stays replicated outside the collective
(adapters/fsdp_islands.py). No Note may say otherwise. CPU only.
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.adapters import fsdp as fsdp_adapter
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.fsdp_islands import OUTSIDE_FORWARD_MODULE_ATTRS
from dgx_monarch.adapters.fsdp_lora_admission import refuse_unless_fsdp_lora_admits
from dgx_monarch.nodes import auto_gate, common, samplers
from dgx_monarch.studio.inventory import load_inventory

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = sorted((ROOT / "example_workflows").glob("*.json"))
sys.path.insert(0, str(ROOT / "tools"))
try:
    import gen_templates
finally:
    sys.path.remove(str(ROOT / "tools"))

# The request kind each sampler node sends to run_render; the pin below reads
# it from each node's source.
REQUEST_KIND = {"DGXMonarchKSampler": "ksampler",
                "DGXMonarchKSamplerAdvanced": "ksampler_advanced",
                "DGXMonarchSamplerCustom": "custom"}


def _doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _note(doc: dict) -> str:
    return " ".join(str(value) for node in doc["nodes"] if node["type"] == "Note"
                    for value in node.get("widgets_values") or [])


def _samplers(doc: dict) -> set[str]:
    return {node["type"] for node in doc["nodes"]} & set(REQUEST_KIND)


def _dual(doc: dict) -> bool:
    return any(node["type"] == "DGXMonarchUncondUNETLoader" for node in doc["nodes"])


def test_each_sampler_node_sends_the_request_kind_this_table_names():
    for name, kind in REQUEST_KIND.items():
        source = inspect.getsource(getattr(samplers, name).sample)
        assert f'"kind": "{kind}"' in source, name


def _gate_rig(monkeypatch):
    """A first_use Init whose levers are risky; ensure_live marks that the kind check passed."""
    class PastTheKindCheck(Exception):
        pass

    def reached(_handle):
        raise PastTheKindCheck

    monkeypatch.setattr(common, "ensure_live", reached)
    mesh = SimpleNamespace(auto_gate="first_use", handle=object(), worker_args={})
    return SimpleNamespace(mesh=mesh, loras=({"name": "lora"},), unet_name="m.safetensors"), PastTheKindCheck


def test_a_custom_sampler_request_never_reaches_the_first_use_gate(monkeypatch):
    model, past = _gate_rig(monkeypatch)
    assert auto_gate.auto_gate_context(model, "custom") is None
    for kind in ("ksampler", "ksampler_advanced"):
        with pytest.raises(past):
            auto_gate.auto_gate_context(model, kind)


def test_a_dual_model_request_runs_no_ceremony_and_renders_stock(monkeypatch):
    model, _past = _gate_rig(monkeypatch)
    request = {"kind": "custom", "uncond_model": {"unet_name": "u.safetensors"}}
    assert auto_gate.maybe_auto_gate(model, request, {}, 7.0, 20) == "DUAL_MODEL_STOCK"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.stem)
def test_a_note_states_the_gate_its_own_sampler_gets(path):
    doc = _doc(path)
    note, kinds = _note(doc), {REQUEST_KIND[name] for name in _samplers(doc)}
    if _dual(doc):
        assert gen_templates.GATE_DUAL in note
        assert gen_templates.GATE not in note and gen_templates.GATE_CUSTOM not in note
    elif kinds == {"custom"}:
        assert gen_templates.GATE_CUSTOM in note
        assert gen_templates.GATE not in note
    else:
        assert gen_templates.GATE_CUSTOM not in note and gen_templates.GATE_DUAL not in note
    # The automatic gate is promised only where a KSampler sends the render.
    if "runs the identity gate" in note or gen_templates.GATE in note:
        assert kinds & {"ksampler", "ksampler_advanced"} and not _dual(doc)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.stem)
def test_no_note_describes_the_gate_as_a_single_gpu_comparison_or_unbounded(path):
    note = _note(_doc(path))
    assert "single-GPU load" not in note
    assert not re.search(r"as conclusive(ly)? as", note)


def test_the_gate_node_tooltip_keeps_the_check_finite():
    # The generated inventory reads the tooltip from the node's source, as Studio shows it.
    (row,) = [row for row in load_inventory()["rows"]
              if row["id"] == "node.input.DGXMonarchIdentityGate.steps"]
    assert row["default"] == 2
    assert "conclusive" not in row["applicability"]
    assert "finite check" in row["applicability"]


def test_the_fsdp_precision_sentence_matches_the_launch_contract():
    sentence = gen_templates.FSDP_RULES
    for kind in ("bf16", "fp16", "fp8", "int8"):
        assert kind in sentence
        fsdp_adapter.validate_fsdp_launch_quant(kind)
    for kind in ("mxfp8", "nvfp4"):
        assert kind in sentence
        with pytest.raises(UnsupportedModelError):
            fsdp_adapter.validate_fsdp_launch_quant(kind)
    assert "fp8 weight_dtype" in sentence
    with pytest.raises(UnsupportedModelError):
        fsdp_adapter.refuse_fsdp_live_cast("fp8_e4m3fn")
    # The LoRA half: low-RSS on, and no comfy-kitchen quantized file.
    assert "lora_low_rss" in sentence
    with pytest.raises(UnsupportedModelError):
        fsdp_adapter.validate_fsdp_launch_loras([{"name": "l"}], lora_low_rss=False)
    with pytest.raises(UnsupportedModelError):
        refuse_unless_fsdp_lora_admits(SimpleNamespace(quantized_shards=True))
    refuse_unless_fsdp_lora_admits(SimpleNamespace(quantized_shards=False))


# Sentences an earlier launch contract made true. Each now contradicts the code.
_STALE_FSDP = (
    r"refuses every\s+quantized file",
    r"takes bf16 only",
    r"refuses fp8, int8",
    r"FP8 FSDP\s+each refuse",
    r"FP8\s+with FSDP is out of scope",
    r"llm_adapter never joins",
    r"FSDP preset typed-refuses",
)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.stem)
def test_no_note_keeps_a_retired_fsdp_refusal(path):
    note = _note(_doc(path))
    for pattern in _STALE_FSDP:
        assert not re.search(pattern, note), (path.stem, pattern)


def test_anima_keeps_its_adapter_replicated_and_its_note_says_so():
    assert "llm_adapter" in OUTSIDE_FORWARD_MODULE_ATTRS["Anima"]
    note = _note(_doc(ROOT / "example_workflows" / "dgx-monarch-anima-t2i.json"))
    assert "llm_adapter" in note and "replicated" in note
    assert gen_templates.FSDP_RULES in note


@pytest.mark.parametrize("name", ["dgx-monarch-flux2-t2i", "dgx-monarch-kandinsky5-video-pro",
                                  "dgx-monarch-anima-t2i"])
def test_the_fsdp_guidance_notes_carry_the_launch_rules(name):
    assert gen_templates.FSDP_RULES in _note(_doc(ROOT / "example_workflows" / f"{name}.json"))
