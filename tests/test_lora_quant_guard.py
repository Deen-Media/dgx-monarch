"""A LoRA on a MOSS-Image INT8 file refuses class P on every topology (#129).

``lora_quant_guard`` holds the one predicate and the one refusal. The driver
reads the checkpoint header before dispatch, for the model and for the
unconditional model of a two-checkpoint guider, in the per-render preflight and
in Fleet KSampler. Every Worker's model store refuses the same pair before a
lazy or a hot swap on a resident base and before the LoRA build of a fresh
load. A render without a LoRA, a BF16 file with one, and another family's INT8
file with one stay open at every site.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import lora_quant_guard
from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import fleet as fleet_mod
from dgx_monarch.nodes import model_patch_preflight as preflight
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.topology import topology_from_preset
from store_ensure_helpers import STACK
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.

LORA = [{"name": "dmd.safetensors", "strength": 1.0}]
TROUBLESHOOTING = Path(__file__).resolve().parents[1] / "docs" / "TROUBLESHOOTING.md"


def _class_p(exc: BaseException) -> str:
    text = str(exc)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS, text
    assert tag.guard is None and not tag.waivable, text
    assert text.endswith("See docs/TROUBLESHOOTING.md #129."), text
    return text


# --- the predicate and the refusal ----------------------------------------------

def test_only_a_moss_image_int8_file_with_a_lora_refuses():
    assert lora_quant_guard.lora_refused("moss_image", "int8", LORA)
    for family, quant, stack in (
        ("moss_image", "int8", None), ("moss_image", "int8", []),
        ("moss_image", "bf16", LORA), ("moss_image", "nvfp4", LORA),
        ("flux3", "int8", LORA), ("krea2", "int8", LORA), (None, "int8", LORA),
        ("moss_image", None, LORA),
    ):
        assert not lora_quant_guard.lora_refused(family, quant, stack), (family, quant, stack)
        lora_quant_guard.assert_lora_quant_supported(family, quant, stack)


def test_the_refusal_names_the_lora_the_measurement_date_and_the_remedy():
    with pytest.raises(UnsupportedModelError) as caught:
        lora_quant_guard.assert_lora_quant_supported("moss_image", "int8", LORA + STACK)
    text = _class_p(caught.value)
    assert text.startswith("[dgxm:P] moss_image: an INT8 file with a LoRA "
                           "(dmd.safetensors, a.safetensors) rendered far from stock"), text
    assert "on 2026-10-21" in text and "every topology" in text
    assert "mode=local on one GPU included" in text
    assert "Render this graph in stock ComfyUI on one GPU instead" in text
    assert "a BF16 file with a LoRA is not refused" in text


def test_the_refusal_names_the_classic_bake_and_the_entry_quotes_it_word_for_word():
    """The #129 Symptom quotes the message around the LoRA names, so a reworded message moves both."""
    with pytest.raises(UnsupportedModelError) as caught:
        lora_quant_guard.assert_lora_quant_supported("moss_image", "int8", LORA)
    text = _class_p(caught.value)
    assert "under investigation" not in text
    assert "dgx-monarch bakes the LoRA into the quantized weights as stock's classic loader does" in text
    assert "Render this graph in stock ComfyUI on one GPU instead, with its default DynamicVRAM loader" in text
    entry = TROUBLESHOOTING.read_text().split("\n## 129. ", 1)[1].split("\n## ", 1)[0]
    symptom = " ".join(entry.split("**Symptom:**", 1)[1].split("**Cause:**", 1)[0].split())
    opener, tail = re.findall(r'"([^"]+)"', symptom)
    assert text.split("] ", 1)[1] == f"{opener} (dmd.safetensors) {tail}"


# --- the driver -----------------------------------------------------------------

def _model(*, loras=(), unet_name="kromoss_int8_convrot.safetensors") -> ModelSpec:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    return ModelSpec(mesh=mesh, unet_name=unet_name, options={"weight_dtype": "default"},
                     loras=tuple(loras))


@pytest.fixture
def sniff(monkeypatch):
    """The header sniff, swappable per test, counting its reads."""
    state = {"result": ("moss_image", "int8"), "names": []}

    def fake(unet_name, override):
        state["names"].append(unet_name)
        return state["result"]

    monkeypatch.setattr(preflight, "_sniffed_checkpoint", fake)
    return state


@pytest.mark.parametrize("preset,world", [
    ("single", 1), ("dp2", 2), ("uly2", 2), ("cfg2", 2), ("ring2", 2), ("uly2+fsdp", 2),
])
def test_the_driver_refuses_before_dispatch_on_every_topology(sniff, preset, world):
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_model_patches(_model(loras=LORA), topology_from_preset(preset, world), {})
    _class_p(caught.value)
    assert sniff["names"] == ["kromoss_int8_convrot.safetensors"]


def test_the_driver_reads_no_header_without_a_lora(sniff):
    preflight.preflight_model_patches(_model(), topology_from_preset("uly2", 2), {})
    preflight.preflight_lora_quant(_model(), {"unet_name": "other.safetensors", "loras": []})
    assert sniff["names"] == []


@pytest.mark.parametrize("result", [("moss_image", "bf16"), ("flux3", "int8"), None],
                         ids=["moss-bf16", "flux3-int8", "unreadable-header"])
def test_the_driver_lets_a_bf16_file_another_family_and_an_unreadable_header_through(
        sniff, result):
    sniff["result"] = result
    preflight.preflight_model_patches(_model(loras=LORA), topology_from_preset("uly2", 2), {})
    assert sniff["names"] == ["kromoss_int8_convrot.safetensors"]


def test_the_driver_checks_a_two_checkpoint_guiders_unconditional_model(sniff):
    uncond = {"unet_name": "kromoss_int8_convrot.safetensors", "loras": list(LORA)}
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_model_patches(
            _model(unet_name="kromoss_bf16.safetensors"), topology_from_preset("dp2", 2),
            {"uncond_model": uncond})
    _class_p(caught.value)
    assert sniff["names"] == ["kromoss_int8_convrot.safetensors"]


def test_fleet_ksampler_refuses_before_its_first_dispatch(sniff, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Fleet KSampler dispatched a refused LoRA pair")

    monkeypatch.setattr(fleet_mod.DGXMonarchFleetKSampler, "_fleet_bound", unexpected)
    with pytest.raises(UnsupportedModelError) as caught:
        fleet_mod.DGXMonarchFleetKSampler().fleet(
            [_model(loras=LORA)], [object()], ["NEG"], [{"samples": torch.zeros(1)}],
            [""], [42], [1], [1.0], ["euler"], ["simple"], positive=["POS"])
    _class_p(caught.value)


# --- the Worker belt ------------------------------------------------------------

def _detects(monkeypatch, family: str, quant: str) -> None:
    monkeypatch.setattr(ms, "_detect_family", lambda _patcher: family)
    monkeypatch.setattr(ms, "_detect_live_precision",
                        lambda _patcher, _kind: LivePrecisionEvidence(quant, f"all_{quant}"))


def _counting_build(monkeypatch) -> list:
    builds: list = []

    def build(self, base, stack):
        builds.append(list(stack or []))
        return SimpleNamespace(tag="ACTIVE", model=base.model)

    monkeypatch.setattr(ms.ModelStore, "_build_active", build)
    return builds


def test_a_fresh_load_refuses_before_the_lora_build_and_leaves_the_slot_empty(rig, monkeypatch):
    store, calls = rig
    _detects(monkeypatch, "moss_image", "int8")
    builds = _counting_build(monkeypatch)
    with pytest.raises(UnsupportedModelError) as caught:
        store.ensure("kromoss_int8_convrot.safetensors", {}, list(LORA))
    _class_p(caught.value)
    assert calls.load == 1 and builds == []
    assert store.current is None and "cond" not in store._drop_failed


@pytest.mark.parametrize("family,quant,stack", [
    ("moss_image", "int8", None), ("moss_image", "bf16", LORA), ("flux3", "int8", LORA),
], ids=["moss-int8-no-lora", "moss-bf16-lora", "flux3-int8-lora"])
def test_a_fresh_load_stays_open_without_the_refused_pair(rig, monkeypatch, family, quant, stack):
    store, _calls = rig
    _detects(monkeypatch, family, quant)
    builds = _counting_build(monkeypatch)
    store.ensure("model.safetensors", {}, list(stack) if stack else None)
    assert builds == [list(stack or [])]
    assert store.current is not None and store.current.quant_kind == quant


@pytest.mark.parametrize("low_rss", [True, False], ids=["lazy-swap", "hot-swap"])
def test_a_resident_base_refuses_a_lora_before_either_swap(rig, monkeypatch, low_rss):
    store, _calls = rig
    store.lora_low_rss = low_rss
    _detects(monkeypatch, "moss_image", "int8")
    store.ensure("kromoss_int8_convrot.safetensors", {}, None)
    resident = store.current
    assert resident is not None and resident.family == "moss_image" and resident.quant_kind == "int8"

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a swap ran for a refused LoRA pair")

    monkeypatch.setattr(ms.store_bake, "lazy_swap_transition", unexpected)
    monkeypatch.setattr(ms.store_bake, "hot_swap_transition", unexpected)
    monkeypatch.setattr(ms.store_load, "fresh_load", unexpected)
    with pytest.raises(UnsupportedModelError) as caught:
        store.ensure("kromoss_int8_convrot.safetensors", {}, list(LORA))
    _class_p(caught.value)
    assert store.current is resident and "cond" not in store._drop_failed


@pytest.mark.parametrize("low_rss", [True, False], ids=["lazy-swap", "hot-swap"])
def test_a_resident_bf16_base_still_swaps_a_lora(rig, monkeypatch, low_rss):
    store, _calls = rig
    store.lora_low_rss = low_rss
    _detects(monkeypatch, "moss_image", "bf16")
    store.ensure("kromoss_bf16.safetensors", {}, None)
    swaps: list[str] = []

    def swap(name):
        def transition(*_args, **_kwargs):
            swaps.append(name)
            return "ACTIVE", "hot-swap"
        return transition

    monkeypatch.setattr(ms.store_bake, "lazy_swap_transition", swap("lazy"))
    monkeypatch.setattr(ms.store_bake, "hot_swap_transition", swap("hot"))
    assert store.ensure("kromoss_bf16.safetensors", {}, list(LORA)) == ("ACTIVE", "hot-swap")
    assert swaps == ["lazy" if low_rss else "hot"]
