"""What a template Note says about the identity gate and FSDP matches the code.

The gate: the first-use gate compares renders on the graph's own mesh and
topology, never a stock single-GPU load (nodes/gate_ceremony.py,
nodes/gate_cross_mode.py). Which renders reach it is set by the two switches
below, AUTO_GATED_KINDS and DUAL_MODEL_AUTO_GATES: since 1.5 every sampler
kind does (auto_gate.auto_gate_context), and a dual-model render proves both of
its models (nodes/gate_dual.py). Each graph's Note must say the sentence for its
own sampler.

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


# ---- Which renders reach the first-use gate -------------------------------
# These two switches hold what the runtime gates, and every claim in this
# section reads them. Since 1.5 both are on: the code pins require the gate to
# reach Sampler Custom and dual-model requests, the eight graphs those used to
# leave out carry GATE, the Identity Gate Note has no Sampler Custom sentence,
# and MODELS.md and TROUBLESHOOTING.md hold none of the lines below.
AUTO_GATED_KINDS = frozenset({"ksampler", "ksampler_advanced", "custom"})
DUAL_MODEL_AUTO_GATES = True
# Doc sentences that hold only while a switch is off, with the switch they follow.
NO_GATE_SENTENCES = (
    ("custom", "example_workflows/dgx-monarch-identity-gate.json",
     "Sampler Custom graphs never do, so this node is how they earn slab and low-RSS residency"),
    ("custom", "docs/TROUBLESHOOTING.md",
     "Single-model custom samplers require the explicit Identity Gate node."),
    ("dual", "docs/TROUBLESHOOTING.md",
     "Dual-model custom requests are always dispatched with both residency optimizations off"),
    ("dual", "docs/MODELS.md", "This dual-model graph runs no identity gate"),
)


def _switch_on(name: str) -> bool:
    return "custom" in AUTO_GATED_KINDS if name == "custom" else DUAL_MODEL_AUTO_GATES


def _auto_gated(doc: dict) -> bool:
    """Whether a render of this graph reaches the first-use gate under the switches above."""
    kinds = {REQUEST_KIND[name] for name in _samplers(doc)}
    return bool(kinds & AUTO_GATED_KINDS) and (DUAL_MODEL_AUTO_GATES or not _dual(doc))


def _gate_rig(monkeypatch):
    """A first_use Init whose levers are risky; ensure_live marks that the kind check passed."""
    class PastTheKindCheck(Exception):
        pass

    def reached(_handle):
        raise PastTheKindCheck

    monkeypatch.setattr(common, "ensure_live", reached)
    mesh = SimpleNamespace(auto_gate="first_use", handle=object(), worker_args={})
    return SimpleNamespace(mesh=mesh, loras=({"name": "lora"},), unet_name="m.safetensors"), PastTheKindCheck


@pytest.mark.parametrize("kind", sorted(REQUEST_KIND.values()))
def test_the_first_use_gate_reaches_exactly_the_switched_kinds(monkeypatch, kind):
    model, past = _gate_rig(monkeypatch)
    if kind in AUTO_GATED_KINDS:
        with pytest.raises(past):
            auto_gate.auto_gate_context(model, kind)
    else:
        assert auto_gate.auto_gate_context(model, kind) is None


def test_a_dual_model_request_follows_its_switch(monkeypatch):
    model, past = _gate_rig(monkeypatch)
    request = {"kind": "custom", "uncond_model": {"unet_name": "u.safetensors"}}
    try:
        verdict = auto_gate.maybe_auto_gate(model, request, {}, 7.0, 20)
    except past:
        verdict = None  # it went on to read the gate context
    if DUAL_MODEL_AUTO_GATES:
        assert verdict != "DUAL_MODEL_STOCK"
    else:
        assert verdict == "DUAL_MODEL_STOCK"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.stem)
def test_a_note_states_the_gate_its_own_sampler_gets(path):
    doc = _doc(path)
    note = _note(doc)
    custom = getattr(gen_templates, "GATE_CUSTOM", None)
    dual = getattr(gen_templates, "GATE_DUAL", None)
    kinds = {REQUEST_KIND[name] for name in _samplers(doc)}
    if _auto_gated(doc) or not kinds:
        # Gated, or no listed sampler (Fleet, the Identity Gate graph): neither sentence.
        assert not [text for text in (custom, dual) if text and text in note]
    elif _dual(doc):
        assert dual and dual in note
        assert gen_templates.GATE not in note and not (custom and custom in note)
    else:
        assert custom and custom in note
        assert gen_templates.GATE not in note
    # The automatic gate is promised only where a render reaches it.
    if "runs the identity gate" in note or gen_templates.GATE in note:
        assert _auto_gated(doc)


@pytest.mark.parametrize(("switch", "name", "sentence"), NO_GATE_SENTENCES,
                         ids=lambda value: str(value)[:24])
def test_a_no_gate_sentence_stands_only_while_its_switch_is_off(switch, name, sentence):
    text = (ROOT / name).read_text(encoding="utf-8")
    if name.endswith(".json"):
        text = _note(json.loads(text))
    # Blockquote markers would split a sentence that wraps inside one.
    text = " ".join(line.lstrip("> ") for line in text.split("\n"))
    present = sentence in " ".join(text.split())
    assert present is not _switch_on(switch), (name, sentence)


def test_the_dual_graph_names_the_placements_its_proof_cannot_run_on():
    """cfg2 renders stock and *+fsdp refuses under first_use (nodes/gate_dual.py)."""
    from dgx_monarch.nodes import gate_dual
    from dgx_monarch.topology import Topology

    sentence = gen_templates.GATE_DUAL_PLACEMENTS
    assert gate_dual.skip_reason(Topology(cfg=2, world=2), {}) == gate_dual.CFG_PARALLEL_REASON
    assert "cfg2" in sentence and "stock residency" in sentence
    assert gate_dual.skip_reason(Topology(ulysses=2, fsdp=True, world=2), {}) == gate_dual.FSDP_REASON
    assert "*+fsdp preset refuses" in sentence
    assert gate_dual.skip_reason(Topology(ulysses=2, world=2), {}) is None
    for path in SHIPPED:
        doc = _doc(path)
        assert (sentence in _note(doc)) is _dual(doc), path.stem


# ---- What the gate compares --------------------------------------------------
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


@pytest.mark.parametrize("name", ["dgx-monarch-wan-scail", "dgx-monarch-wan21-i2v"])
def test_a_wan_bf16_file_under_fsdp_is_told_to_load_as_bf16(name):
    # ComfyUI runs the Wan 2.1 family in fp16 by default, so a bf16 file loads
    # as fp16 and FSDP's launch check reads that kind, not the file's, and
    # refuses the mismatch (docs/TROUBLESHOOTING.md #17).
    from dgx_monarch.actor.store_detect import _combine_checkpoint_and_live_kinds

    assert _combine_checkpoint_and_live_kinds("bf16", "fp16") == "fp16"
    assert _combine_checkpoint_and_live_kinds("bf16", "bf16") == "bf16"
    note = " ".join(_note(_doc(ROOT / "example_workflows" / f"{name}.json")).split())
    assert re.search(r"weight_dtype (set )?to bf16 on (the )?loader", note)
