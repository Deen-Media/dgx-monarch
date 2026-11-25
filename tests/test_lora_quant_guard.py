"""A LoRA on a MOSS-Image INT8, grouped INT8 or ConvRot W4A4 file refuses class P on every topology (#129).

``lora_quant_guard`` holds the one predicate and the one refusal. The driver
reads the checkpoint header before dispatch, for the model and for the
unconditional model of a two-checkpoint guider, in the per-render preflight and
in Fleet KSampler. Every Worker's model store refuses the same pair before a
lazy or a hot swap on a resident base and before the LoRA build of a fresh
load. A render without a LoRA, a BF16 file with one, and another family's INT8
file with one stay open at every site.
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import lora_quant_guard
from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.adapters import detect, layer_records
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.moss_image import MOSSImageAdapter
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

def test_only_a_moss_image_int8_grouped_or_w4a4_file_with_a_lora_refuses():
    for quant in ("int8", "w4a8", "w4a4"):
        assert lora_quant_guard.lora_refused("moss_image", quant, LORA), quant
    assert lora_quant_guard.LORA_REFUSED_QUANTS == {"moss_image": frozenset({"int8", "w4a8", "w4a4"})}
    for family, quant, stack in (
        ("moss_image", "int8", None), ("moss_image", "int8", []),
        ("moss_image", "w4a8", None), ("moss_image", "w4a4", []),
        ("moss_image", "bf16", LORA), ("moss_image", "nvfp4", LORA),
        ("moss_image", "fp8", LORA), ("moss_image", "mxfp8", LORA),
        ("flux3", "int8", LORA), ("krea2", "int8", LORA), (None, "int8", LORA),
        ("flux3", "w4a8", LORA), ("minimax_h3", "w4a8", LORA), ("krea2", "w4a4", LORA),
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


@pytest.mark.parametrize("quant", ["w4a8", "w4a4"])
def test_an_unmeasured_moss_image_kind_with_a_lora_refuses_in_words_that_state_no_measurement(quant):
    """The grouped INT8 kinds and ConvRot W4A4 on MOSS-Image: no render has
    measured a LoRA on them, and the family's INT8 file with one rendered
    wrong, so the pair refuses everywhere until it is measured. The refusal
    says that, never that this kind was measured, and gives the remedy of #129."""
    assert quant in lora_quant_guard.LORA_UNMEASURED_QUANTS["moss_image"]
    assert lora_quant_guard.LORA_UNMEASURED_QUANTS["moss_image"] < lora_quant_guard.LORA_REFUSED_QUANTS["moss_image"]
    assert "int8" not in lora_quant_guard.LORA_UNMEASURED_QUANTS["moss_image"]
    assert lora_quant_guard.lora_refused("moss_image", quant, LORA)
    for family, stack in (("moss_image", None), ("moss_image", []), ("flux3", LORA), ("minimax_h3", LORA)):
        assert not lora_quant_guard.lora_refused(family, quant, stack), (family, stack)
    with pytest.raises(UnsupportedModelError) as caught:
        lora_quant_guard.assert_lora_quant_supported("moss_image", quant, LORA + STACK)
    text = _class_p(caught.value)
    assert text.startswith(f"[dgxm:P] moss_image: no render has measured a LoRA "
                           f"(dmd.safetensors, a.safetensors) on a {quant.upper()} file of this family. "
                           "Its INT8 file with a LoRA rendered far from stock ComfyUI on the cluster "
                           "on 2026-10-21, and nothing shows this kind does better"), text
    assert "until a render measures it" in text and "every topology" in text
    assert "mode=local on one GPU included" in text
    assert "Render this graph in stock ComfyUI on one GPU instead, with its default DynamicVRAM loader" in text
    assert "a BF16 file with a LoRA is not refused" in text
    assert "dgx-monarch baked the LoRA" not in text and "no hardware run has measured on this file" not in text
    entry = " ".join(TROUBLESHOOTING.read_text().split("\n## 129. ", 1)[1].split("\n## ", 1)[0].split())
    assert "no render has measured a LoRA" in entry and "`w4a8`" in entry and "`w4a4`" in entry


def test_the_refusal_names_the_classic_bake_and_the_entry_quotes_it_word_for_word():
    """The #129 Symptom quotes the message around the LoRA names, so a reworded message moves both."""
    with pytest.raises(UnsupportedModelError) as caught:
        lora_quant_guard.assert_lora_quant_supported("moss_image", "int8", LORA)
    text = _class_p(caught.value)
    assert "under investigation" not in text
    assert "dgx-monarch baked the LoRA into the quantized weights as stock's classic loader does" in text
    assert "The Workers now bake stock's cast-time weight instead (#145)" in text
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
    """The LoRA check reads nothing without a LoRA. On a layout that shards the
    tokens or the cfg batch the sharded-kind check reads the header once."""
    preflight.preflight_model_patches(_model(), topology_from_preset("dp2", 2), {})
    preflight.preflight_lora_quant(_model(), {"unet_name": "other.safetensors", "loras": []})
    assert sniff["names"] == []
    preflight.preflight_model_patches(_model(), topology_from_preset("uly2", 2), {})
    assert sniff["names"] == ["kromoss_int8_convrot.safetensors"]


@pytest.mark.parametrize("result", [("moss_image", "bf16"), ("flux3", "int8"), None],
                         ids=["moss-bf16", "flux3-int8", "unreadable-header"])
def test_the_driver_lets_a_bf16_file_another_family_and_an_unreadable_header_through(
        sniff, result):
    sniff["result"] = result
    preflight.preflight_model_patches(_model(loras=LORA), topology_from_preset("uly2", 2), {})
    # Once for the LoRA check and once for the sharded-kind check of this layout.
    assert sniff["names"] == ["kromoss_int8_convrot.safetensors"] * 2


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


# --- the driver, on real files: an INT8 header that is not an INT8 file ---------

def _moss_file(tmp_path, name: str, formats: tuple[str | None, ...], dtype=torch.int8) -> str:
    """A tiny MOSS-Image checkpoint whose header reads int8, one quantized layer a format.

    Each layer stores an I8 weight beside a ``weight_scale``, as an INT8
    ConvRot layer and a ConvRot W4A4 layer both do, and a ``comfy_quant``
    record naming its format, or none for a format of None. With an FP8
    ``dtype`` the header reads fp8 instead, as an FP8, an MXFP8 and an NVFP4
    file's all do.
    """
    from safetensors.torch import save_file

    state = {"shared_block_modulation.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}
    for index, fmt in enumerate(formats):
        layer = f"layers.{index}.proj.ffn_up_gate"
        state[f"{layer}.weight"] = torch.zeros(2, 2, dtype=dtype)
        state[f"{layer}.weight_scale"] = torch.ones(2, 1)
        if fmt is not None:
            record = json.dumps({"format": fmt}).encode()
            state[f"{layer}.comfy_quant"] = torch.tensor(list(record), dtype=torch.uint8)
    path = tmp_path / name
    save_file(state, str(path))
    return str(path)


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Three files with the same int8 header: INT8 ConvRot, ConvRot W4A4 behind an INT8 layer, and one with no record."""
    paths = {
        "int8.safetensors": _moss_file(tmp_path, "int8.safetensors", ("int8_tensorwise",) * 2),
        "w4a4.safetensors": _moss_file(tmp_path, "w4a4.safetensors", ("int8_tensorwise", "convrot_w4a4")),
        "bare.safetensors": _moss_file(tmp_path, "bare.safetensors", (None, None)),
    }
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, name: paths.get(name)
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    for name, path in paths.items():
        assert detect.sniff_checkpoint(path) == ("moss_image", "int8"), name
    return paths


@pytest.fixture
def sniff_real(monkeypatch):
    """The real header sniff, with an answer a test may put in its place: ``result`` for every
    file, ``by_name`` for one. None, the default, reads the file."""
    state: dict = {"result": None, "by_name": {}}
    real = preflight._sniffed_checkpoint

    def sniffed(unet_name, override):
        if unet_name in state["by_name"]:
            return state["by_name"][unet_name]
        return state["result"] if state["result"] is not None else real(unet_name, override)

    monkeypatch.setattr(preflight, "_sniffed_checkpoint", sniffed)
    return state


def _driver_refusal(unet_name: str, preset: str = "single", world: int = 1) -> str:
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_model_patches(
            _model(loras=LORA, unet_name=unet_name), topology_from_preset(preset, world), {})
    return _class_p(caught.value)


def test_the_driver_names_an_int8_header_by_the_files_own_layer_records(files):
    assert layer_records.name_int8_header_kind(files["int8.safetensors"]) == "int8"
    assert layer_records.name_int8_header_kind(files["w4a4.safetensors"]) == "w4a4"
    assert layer_records.name_int8_header_kind(files["bare.safetensors"]) is None
    assert layer_records.sniff_layer_formats(files["w4a4.safetensors"]) == {"int8_tensorwise", "convrot_w4a4"}
    assert layer_records.name_int8_header_kind(files["int8.safetensors"] + ".gone") is None


def test_a_file_swapped_between_the_look_and_the_read_is_filed_under_its_own_identity(
        files, tmp_path, monkeypatch):
    """The path is looked at, then opened. A file put in its place between the
    two must be remembered as itself: the answer is kept under the identity of
    the descriptor that was read, so the file it replaced still reads as
    what it is, and one open serves the header and the records alike."""
    int8, w4a4 = files["int8.safetensors"], files["w4a4.safetensors"]
    link = tmp_path / "model.safetensors"
    link.symlink_to(int8)
    monkeypatch.setattr(layer_records, "_FORMATS_BY_IDENTITY", type(layer_records._FORMATS_BY_IDENTITY)())
    real_open, opens = layer_records._open_readonly, []

    def swap_then_open(path):
        if not opens:  # after the look at the INT8 file, before the first open
            link.unlink()
            link.symlink_to(w4a4)
        opens.append(str(path))
        return real_open(path)

    monkeypatch.setattr(layer_records, "_open_readonly", swap_then_open)
    assert layer_records.name_int8_header_kind(link) == "w4a4"  # the file that was read
    assert len(opens) == 1
    link.unlink()
    link.symlink_to(int8)
    # The INT8 file is the one the first look saw. It was never read, so nothing is filed under it.
    assert layer_records.name_int8_header_kind(link) == "int8"
    assert layer_records.name_int8_header_kind(int8) == "int8"
    assert layer_records.name_int8_header_kind(w4a4) == "w4a4"
    assert len(opens) == 2, "a file already read is answered with no open"
    kept = dict(layer_records._FORMATS_BY_IDENTITY)
    assert len(kept) == 2 and sorted(map(sorted, kept.values())) == [
        ["convrot_w4a4", "int8_tensorwise"], ["int8_tensorwise"]]


def test_a_file_that_changes_while_it_is_read_is_not_kept(files, monkeypatch):
    monkeypatch.setattr(layer_records, "_FORMATS_BY_IDENTITY", type(layer_records._FORMATS_BY_IDENTITY)())
    real = layer_records._records_of

    def touched_meanwhile(fd, identity):
        formats = real(fd, identity)
        os.utime(files["int8.safetensors"], ns=(1, 1))
        return formats

    monkeypatch.setattr(layer_records, "_records_of", touched_meanwhile)
    assert layer_records.sniff_layer_formats(files["int8.safetensors"]) is None
    assert not layer_records._FORMATS_BY_IDENTITY
    monkeypatch.setattr(layer_records, "_records_of", real)
    assert layer_records.sniff_layer_formats(files["int8.safetensors"]) == {"int8_tensorwise"}


@pytest.mark.parametrize("preset,world", [("single", 1), ("dp2", 2), ("uly2", 2), ("cfg2", 2)])
def test_the_drivers_lora_refusal_describes_the_file_it_holds(files, preset, world):
    """An INT8 file gets the words of its measurement, unchanged. A ConvRot W4A4
    file, whose header reads int8 too, gets the unmeasured words that name it.
    A file with no layer record gets words true of both, never INT8 alone."""
    with pytest.raises(UnsupportedModelError) as caught:
        lora_quant_guard.assert_lora_quant_supported("moss_image", "int8", LORA)
    measured = _class_p(caught.value)
    assert _driver_refusal("int8.safetensors", preset, world) == measured

    w4a4 = _driver_refusal("w4a4.safetensors", "single", 1)
    with pytest.raises(UnsupportedModelError) as caught:
        lora_quant_guard.assert_lora_quant_supported("moss_image", "w4a4", LORA)
    assert w4a4 == _class_p(caught.value)
    assert "no render has measured a LoRA (dmd.safetensors) on a W4A4 file" in w4a4
    assert "an INT8 file with a LoRA (dmd.safetensors) rendered far" not in w4a4

    bare = _driver_refusal("bare.safetensors", preset, world)
    assert bare.startswith("[dgxm:P] moss_image: an INT8 or ConvRot W4A4 file with a LoRA "
                           "(dmd.safetensors) is refused. Its header reads int8, as both kinds do"), bare
    assert "This family's INT8 file with a LoRA rendered far from stock" in bare
    assert "no render has measured a ConvRot W4A4 file with one" in bare
    assert "an INT8 file with a LoRA (dmd.safetensors) rendered far" not in bare
    assert lora_quant_guard.WORKING_ALTERNATIVE in bare and lora_quant_guard.WORKING_ALTERNATIVE in w4a4
    entry = " ".join(TROUBLESHOOTING.read_text().split("\n## 129. ", 1)[1].split("\n## ", 1)[0].split())
    assert "moss_image: an INT8 or ConvRot W4A4 file with a LoRA" in entry


def test_the_both_kinds_words_are_the_drivers_alone(files):
    """Only a header's int8 on a family that refuses w4a4 as unmeasured is an open
    question; a Worker has read the layers and never asks for these words."""
    assert lora_quant_guard.header_kind_is_open("moss_image", "int8")
    for family, kind in (("moss_image", "w4a8"), ("moss_image", "w4a4"), ("moss_image", "bf16"),
                         ("flux3", "int8"), (None, "int8")):
        assert not lora_quant_guard.header_kind_is_open(family, kind), (family, kind)
    with pytest.raises(UnsupportedModelError) as caught:
        lora_quant_guard.assert_lora_quant_supported("moss_image", "w4a8", LORA, int8_or_w4a4=True)
    assert "no render has measured a LoRA (dmd.safetensors) on a W4A8 file" in str(caught.value)
    lora_quant_guard.assert_lora_quant_supported("flux3", "int8", LORA, int8_or_w4a4=True)


@pytest.mark.parametrize("preset", ["uly2", "ring2", "cfg2"])
def test_the_driver_refuses_an_unmeasured_kind_on_a_sharded_layout_before_dispatch(files, sniff_real, preset):
    """The Worker's own check, on what the driver can read without a load: a
    ConvRot W4A4 file by its layer records, a grouped or FP8 header by the
    header, an fp8 weight_dtype cast by the loader's option."""
    topo = topology_from_preset(preset, 2)
    adapter = MOSSImageAdapter()

    def refused(model, source: str, kind: str) -> None:
        with pytest.raises(UnsupportedModelError) as caught:
            preflight.preflight_model_patches(model, topo, {})
        text = str(caught.value)
        tag = parse_leading_refusal_tag(text)
        assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS, text
        assert text.startswith(f"[dgxm:P] moss_image: {source} reads as quant={kind}, and on a layout "
                               "that shards the tokens or the cfg batch this family admits only "
                               "bf16, int8. "), text
        assert adapter.sp_quant_refusal_reason in text
        assert "Use dp2 with an even batch, or mode=local with gpus_per_host=1" in text

    refused(_model(unet_name="w4a4.safetensors"), "a layer record of the file", "w4a4")
    sniff_real["result"] = ("moss_image", "w4a8")
    refused(_model(unet_name="int8.safetensors"), "the file's header", "w4a8")
    sniff_real["result"] = ("moss_image", "bf16")
    cast = ModelSpec(mesh=_model().mesh, unet_name="int8.safetensors",
                     options={"weight_dtype": "fp8_e4m3fn"}, loras=())
    refused(cast, "the file under the loader's fp8 weight_dtype", "fp8")
    # What the family measured, what the driver cannot name, and another family all pass.
    sniff_real["result"] = None
    for name in ("int8.safetensors", "bare.safetensors"):
        preflight.preflight_model_patches(_model(unet_name=name), topo, {})
    for result in (("moss_image", "bf16"), ("moss_image", "unknown"), ("moss_image", "fp16"),
                   ("minimax_h3", "w4a8"), ("flux3", "fp8")):
        sniff_real["result"] = result
        preflight.preflight_model_patches(_model(unet_name="int8.safetensors"), topo, {})


_FP8_HEADER_FILES = {
    "fp8.safetensors": (("float8_e4m3fn", "float8_e4m3fn"), "fp8"),
    "fp8_e5m2.safetensors": (("float8_e5m2", "float8_e4m3fn"), "fp8"),
    "mxfp8.safetensors": (("mxfp8", "mxfp8"), "mxfp8"),
    "nvfp4.safetensors": (("nvfp4", "nvfp4"), "nvfp4"),
    # One kind cannot be named: a Worker names a mixed file by its first
    # quantized layer, an order the header does not hold.
    "nvfp4_mixed.safetensors": (("float8_e4m3fn", "nvfp4"), None),
    "other_format.safetensors": (("nvfp4", "int8_tensorwise"), None),
    "plain_fp8.safetensors": ((None, None), None),
}


@pytest.fixture
def fp8_files(tmp_path, monkeypatch):
    """Files whose header reads fp8 alike: FP8, MXFP8 and NVFP4 by their layer records, and three that do not say."""
    paths = {name: _moss_file(tmp_path, name, formats, dtype=torch.float8_e4m3fn)
             for name, (formats, _kind) in _FP8_HEADER_FILES.items()}
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, name: paths.get(name)
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    for name, path in paths.items():
        assert detect.sniff_checkpoint(path) == ("moss_image", "fp8"), name
    return paths


def test_the_driver_names_an_fp8_header_by_the_files_own_layer_records(fp8_files):
    for name, (_formats, kind) in _FP8_HEADER_FILES.items():
        assert layer_records.name_fp8_header_kind(fp8_files[name]) == kind, name
    assert layer_records.name_fp8_header_kind(fp8_files["fp8.safetensors"] + ".gone") is None


@pytest.mark.parametrize("preset", ["uly2", "cfg2"])
@pytest.mark.parametrize("name", sorted(_FP8_HEADER_FILES))
def test_the_drivers_sharded_refusal_names_the_kind_behind_an_fp8_header(fp8_files, name, preset):
    """A header reads fp8 for an FP8, an MXFP8 and an NVFP4 file alike, and the
    driver's refusal now comes first, so it has to say which the file is: by
    the layer records where every one names the same kind, and otherwise in
    words true of all three, never fp8 alone."""
    kind = _FP8_HEADER_FILES[name][1]
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_model_patches(_model(unet_name=name), topology_from_preset(preset, 2), {})
    text = str(caught.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS, text
    tail = (", and on a layout that shards the tokens or the cfg batch this family admits only "
            "bf16, int8. ")
    if kind is None:
        assert text.startswith(
            "[dgxm:P] moss_image: the file's header reads as quant=fp8, as the header of an FP8, "
            "an MXFP8 and an NVFP4 file all do" + tail), text
    else:
        assert text.startswith(
            f"[dgxm:P] moss_image: the file by its layer records reads as quant={kind}" + tail), text
    assert MOSSImageAdapter().sp_quant_refusal_reason in text
    assert "Use dp2 with an even batch, or mode=local with gpus_per_host=1" in text
    # One GPU and dp2 shard nothing, so no kind is refused there.
    for open_preset, world in (("single", 1), ("dp2", 2)):
        preflight.preflight_model_patches(_model(unet_name=name), topology_from_preset(open_preset, world), {})


@pytest.mark.parametrize("preset,world", [("single", 1), ("dp2", 2)])
def test_the_driver_leaves_an_unmeasured_kind_alone_where_nothing_is_sharded(files, sniff_real, preset, world):
    topo = topology_from_preset(preset, world)
    preflight.preflight_model_patches(_model(unet_name="w4a4.safetensors"), topo, {})
    sniff_real["result"] = ("moss_image", "fp8")
    preflight.preflight_model_patches(_model(unet_name="int8.safetensors"), topo, {})


def test_the_driver_checks_the_unconditional_models_kind_on_a_sharded_layout(files, sniff_real):
    uncond = {"unet_name": "w4a4.safetensors", "loras": []}
    sniff_real["by_name"] = {"int8.safetensors": ("moss_image", "bf16")}
    with pytest.raises(UnsupportedModelError, match="a layer record of the file reads as quant=w4a4"):
        preflight.preflight_model_patches(
            _model(unet_name="int8.safetensors"), topology_from_preset("uly2", 2), {"uncond_model": uncond})


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
