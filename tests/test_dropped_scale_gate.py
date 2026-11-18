"""The dropped-scale header gate: upstream_gate.refusal_for, TROUBLESHOOTING #120.

ComfyUI reads a layer's quantization from a ``<layer>.comfy_quant`` record:
a tensor in the file, one written from the ``layers`` table of the header's
``_quantization_metadata``, or one its legacy ``scaled_fp8`` conversion
writes for each ``scale_weight`` it renames (``comfy.utils.convert_old_quants``).
The gate replays, on key names, the conversions and the one prefix strip of
``comfy.sd.load_diffusion_model_state_dict``. A ``weight_scale`` beside an
F8, F4, I8 or U8 weight that ends with no record, or a ``scale_weight`` left
unrenamed, reaches a plain Linear as an unexpected key and is dropped: an fp8
or int8 file then renders with its scales ignored, and the gate refuses it
class P at both funnels before anything loads. The real-ComfyUI cases at the
end check the replay against ComfyUI's own load, key for key.

U8 is examined because of the real-ComfyUI case at the end of this file:
under DynamicVRAM ComfyUI's lazy Linear load assigns a packed 4-bit weight
with no shape check and drops its scale, so the load does not refuse it. The
shipped nvfp4 layout (packed U8 weights, ``weight_scale``, ``weight_scale_2``,
``input_scale``) therefore passes on its ``_quantization_metadata``, and the
same layout with that table removed refuses.

Fixtures are header-only files with real dtypes and shrunk shapes. Every case
also calls ``uncovered_scaled_layers``, which raises on a header it cannot
parse, so a broken fixture cannot pass by reading as None.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    _isolated_process_state,
    _RecordingStore,
)
from dgx_monarch import consent_pending, upstream_gate
from dgx_monarch.actor import store_fsdp
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import loader_preflight
from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader
from dgx_monarch.refusal import RefusalClass, RefusalTag, parse_leading_refusal_tag
from dgx_monarch.safetensors_header import DTYPE_BITS, SafetensorsHeaderError
from module_location_helpers import from_checkout
from stock_prefix_checkouts import MASTER, RELEASE, compiled_only, loaded, package, write_checkout

Spec = dict[str, tuple[str, list[int]]]


def _write_header(path: Path, tensors: Spec, metadata: dict[str, str] | None = None) -> str:
    header: dict[str, object] = {}
    offset = 0
    for key, (dtype, shape) in tensors.items():
        nbytes = math.prod(shape) * DTYPE_BITS[dtype] // 8
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return str(path)


def _layers(*names: str, fmt: str = "nvfp4") -> dict[str, str]:
    return {"_quantization_metadata": json.dumps({"layers": {name: {"format": fmt} for name in names}})}


NORM = {"single_blocks.0.norm.query_norm.scale": ("BF16", [32])}
FP8R = {  # per-row F32 scales, no ComfyUI quant record (the modelopt layout)
    "single_blocks.0.q_proj.weight": ("F8_E4M3", [32, 32]),
    "single_blocks.0.q_proj.weight_scale": ("F32", [32]),
    "single_blocks.0.mlp_out.weight": ("F8_E4M3", [32, 64]),
    "single_blocks.0.mlp_out.weight_scale": ("F32", [32]),
    "img_in.weight": ("BF16", [32, 16]),
    **NORM,
}
INT8 = {  # comfy-quant int8 ConvRot, attention pin included
    "single_blocks.0.qkv.weight": ("I8", [96, 32]),
    "single_blocks.0.qkv.weight_scale": ("F32", [96, 1]),
    "single_blocks.0.qkv.comfy_quant": ("U8", [72]),
    "single_blocks.0.comfy_attention.config": ("U8", [16]),
    **NORM,
}
NVFP4 = {  # the shipped nvfp4 shape: packed U8, an F8 block scale, two F32 scalars
    "single_blocks.0.q_proj.weight": ("U8", [32, 16]),
    "single_blocks.0.q_proj.weight_scale": ("F8_E4M3", [32, 2]),
    "single_blocks.0.q_proj.weight_scale_2": ("F32", []),
    "single_blocks.0.q_proj.input_scale": ("F32", []),
    "txt_mode_blocks.0.q_proj.weight": ("BF16", [32, 32]),
    **NORM,
}
F4 = {
    "single_blocks.0.q_proj.weight": ("F4", [32, 32]),
    "single_blocks.0.q_proj.weight_scale": ("F8_E4M3", [32, 2]),
}
MIXED = {  # one comfy-quant layer, one layer with a scale nothing records
    **INT8,
    "single_blocks.1.mlp_in.weight": ("F8_E4M3", [64, 32]),
    "single_blocks.1.mlp_in.weight_scale": ("F32", [64]),
}
PLAIN_FP8 = {"transformer_blocks.0.attn.to_q.weight": ("F8_E4M3", [32, 32])}  # Qwen-Image
SCALED_FP8 = {  # the legacy layout convert_old_quants rewrites
    "scaled_fp8": ("F8_E4M3", [2]),
    "blocks.0.attn.q.weight": ("F8_E4M3", [32, 32]),
    "blocks.0.attn.q.scale_weight": ("F32", [1]),
    "blocks.0.attn.q.scale_input": ("F32", [1]),
}
INT8_NO_RECORD = {  # an int8 layer and its scale, nothing ComfyUI reads
    "single_blocks.0.qkv.weight": ("I8", [96, 32]),
    "single_blocks.0.qkv.weight_scale": ("F32", [96, 1]),
    **NORM,
}
SCALED_FP8_WEIGHT_SCALE = {  # the legacy converter records scale_weight layers only
    **SCALED_FP8,
    "blocks.0.attn.k.weight": ("F8_E4M3", [32, 32]),
    "blocks.0.attn.k.weight_scale": ("F32", [1]),
}
# ComfyUI strips a prefix only when more than five keys carry it, so every
# prefixed fixture keeps six or more.
PREFIX = "model.diffusion_model."
PREFIXED_NVFP4 = {PREFIX + key: value for key, value in NVFP4.items()}
BF16 = {"single_blocks.0.q_proj.weight": ("BF16", [32, 32]), **NORM}
PAD = {"img_in.weight": ("BF16", [32, 16]), "final_layer.linear.weight": ("BF16", [16, 32])}
PREFIXED_SCALED_FP8 = {PREFIX + key: value for key, value in {**SCALED_FP8, **PAD}.items()}
QUANT_RECORD_ELSEWHERE = {  # the record sits outside the prefix ComfyUI strips
    **{PREFIX + key: value for key, value in {**INT8_NO_RECORD, **PAD,
                                               "txt_in.weight": ("BF16", [32, 32])}.items()},
    "single_blocks.0.qkv.comfy_quant": ("U8", [72]),
}
TEXT_ENCODER_OUTSIDE = {  # ComfyUI's diffusion load filters the encoder away
    **{PREFIX + key: value for key, value in {**BF16, **PAD,
                                               "txt_in.weight": ("BF16", [32, 32]),
                                               "txt_out.weight": ("BF16", [32, 32])}.items()},
    "text_encoders.t5xxl.transformer.encoder.block.0.q.weight": ("F8_E4M3", [32, 32]),
    "text_encoders.t5xxl.transformer.encoder.block.0.q.weight_scale": ("F32", [1]),
}

QM = "single_blocks.0.q_proj"
PASSES = {
    "comfy_quant_int8": (INT8, None),
    "nvfp4_with_its_metadata": (NVFP4, _layers(QM)),
    "f4_with_its_metadata": (F4, _layers(QM)),
    "plain_fp8_no_scales": (PLAIN_FP8, None),
    "legacy_scaled_fp8": (SCALED_FP8, None),
    "prefixed_keys_and_layers": (PREFIXED_NVFP4, _layers(PREFIX + QM)),
    "prefixed_keys_bare_layers": (PREFIXED_NVFP4, _layers(QM)),
    "prefixed_legacy_scaled_fp8": (PREFIXED_SCALED_FP8, None),
    "text_encoder_outside_the_prefix": (TEXT_ENCODER_OUTSIDE, None),
    "bf16": (BF16, {"format": "pt"}),
}
REFUSALS = {
    "fp8r_modelopt": (FP8R, {"quantization": json.dumps({"quant_algo": "FP8"}), "format": "pt"},
                      ("single_blocks.0.mlp_out", QM)),
    "nvfp4_without_metadata": (NVFP4, {"format": "pt"}, (QM,)),
    "f4_without_metadata": (F4, None, (QM,)),
    "mixed_one_uncovered": (MIXED, None, ("single_blocks.1.mlp_in",)),
    "metadata_names_another_layer": (NVFP4, _layers("single_blocks.9.q_proj"), (QM,)),
    # JSON null: ComfyUI reads it as a table that names no layer, and loads on.
    "metadata_is_json_null": (NVFP4, {"_quantization_metadata": "null"}, (QM,)),
    "int8_without_record": (INT8_NO_RECORD, None, ("single_blocks.0.qkv",)),
    "scaled_fp8_with_a_weight_scale": (SCALED_FP8_WEIGHT_SCALE, None, ("blocks.0.attn.k",)),
    # _quantization_metadata turns the legacy conversion off, so scale_weight stays.
    "scaled_fp8_beside_quantization_metadata": (
        SCALED_FP8, _layers("blocks.0.attn.q", fmt="float8_e4m3fn"), ("blocks.0.attn.q",)),
    "metadata_names_under_another_prefix": (
        PREFIXED_NVFP4, _layers("diffusion_model." + QM), (QM,)),
    "quant_record_under_another_prefix": (QUANT_RECORD_ELSEWHERE, None, ("single_blocks.0.qkv",)),
}


@pytest.mark.parametrize("case", sorted(PASSES))
def test_a_file_comfy_reads_whole_passes(tmp_path, case):
    tensors, metadata = PASSES[case]
    path = _write_header(tmp_path / f"{case}.safetensors", tensors, metadata)
    assert upstream_gate.uncovered_scaled_layers(path) == ()
    assert upstream_gate.refusal_for(path) is None


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_scale_comfy_would_drop_refuses_and_names_its_layers(tmp_path, case):
    tensors, metadata, layers = REFUSALS[case]
    path = _write_header(tmp_path / f"{case}.safetensors", tensors, metadata)
    assert upstream_gate.uncovered_scaled_layers(path) == layers
    found = upstream_gate.refusal_for(path)
    assert found is not None
    text, number = found
    assert number == upstream_gate.DROPPED_SCALE_TROUBLESHOOTING == 120
    assert f"stores {len(layers)} quantized layer(s), first {layers[0]}," in text
    assert "[dgxm:" not in text  # the tag is built at the raise


@pytest.mark.parametrize("metadata", [None, {"_quantization_metadata": "{not json"},
                                      {"_quantization_metadata": json.dumps({"format": "x"})}],
                         ids=["no_metadata", "broken_json", "no_layers_table"])
def test_the_l2p_gate_keeps_its_own_text_and_number(tmp_path, metadata):
    """A malformed metadata table cannot turn the #91 refusal off: the L2P
    answer returns before the table is parsed."""
    keys = ["cap_embedder.0.weight", "x_pad_token", "layers.0.attention.qkv.weight",
            "all_x_embedder.16-1.weight", "local_decoder.blocks.0.weight"]
    path = _write_header(tmp_path / "l2p.safetensors", {key: ("BF16", [1]) for key in keys},
                         metadata)
    assert upstream_gate.refusal_for(path) == (upstream_gate.ZIMAGE_L2P_REFUSAL, 91)


def test_an_unreadable_or_foreign_path_answers_none(tmp_path):
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"\xff" * 64)
    with pytest.raises(SafetensorsHeaderError):
        upstream_gate.uncovered_scaled_layers(str(garbage))
    unparsed = _write_header(tmp_path / "broken_metadata.safetensors", NVFP4,
                             {"_quantization_metadata": "{not json"})
    for path in (str(garbage), str(tmp_path / "absent.safetensors"), unparsed,
                 str(tmp_path / "model.ckpt"), "", None):
        assert upstream_gate.refusal_for(path) is None


# Both funnels, through the same fixtures as tests/test_upstream_gated_artifact.py.
@pytest.fixture
def models(tmp_path, monkeypatch):
    _write_header(tmp_path / "fp8r.safetensors", *REFUSALS["fp8r_modelopt"][:2])
    _write_header(tmp_path / "int8.safetensors", INT8)
    _write_header(tmp_path / "nvfp4.safetensors", NVFP4, _layers(QM))

    def get_full_path(_kind, name):
        candidate = tmp_path / str(name)
        return str(candidate) if candidate.exists() else None

    fake = types.ModuleType("folder_paths")
    fake.get_full_path = get_full_path
    fake.get_filename_list = lambda _kind: sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    return tmp_path


def _mesh(**overrides):
    return SimpleNamespace(**{"topology_preset": "dp2", "handle": None, "worker_args": {},
                              **overrides})


def _worker():
    return SimpleNamespace(topology={}, world=1, store=_RecordingStore())


def test_both_funnels_refuse_the_same_class_p_text_before_anything_loads(models, monkeypatch):
    from dgx_monarch.nodes import loaders

    def explode(*_args, **_kwargs):
        raise AssertionError("an answer below the gate ran")

    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology", explode)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", explode)
    monkeypatch.setattr(loaders, "ensure_live", explode)
    with pytest.raises(UnsupportedModelError) as driver:
        DGXMonarchUNETLoader().load(_mesh(), "fp8r.safetensors")
    worker = _worker()
    with pytest.raises(UnsupportedModelError) as funnel:
        store_fsdp.ensure(worker, "fp8r.safetensors", {}, [])
    assert worker.store.calls == [], "the store was entered on a refused artifact"
    text = str(driver.value)
    assert str(funnel.value) == text
    assert parse_leading_refusal_tag(text) == RefusalTag(
        refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)
    assert text.endswith("See docs/TROUBLESHOOTING.md #120.")
    assert "renders with its scales ignored" in text
    assert consent_pending.pending_cards() == []


@pytest.mark.parametrize("name", ["int8.safetensors", "nvfp4.safetensors"])
def test_both_funnels_pass_a_file_comfy_reads_whole(models, name):
    loader_preflight.preflight_upstream_gated_artifact(name)
    worker = _worker()
    store_fsdp.ensure(worker, name, {}, [])
    assert len(worker.store.calls) == 1


# The vendor FLUX.3 files, header reads only, where a ComfyUI checkout stages them.
VENDOR_DITS = tuple(f"flux3_image_klein_{kind}.safetensors" for kind in (
    "bf16", "int8_convrot", "nvfp4", "sd_bf16", "sd_int8_convrot", "sd_nvfp4"))


def _comfy_checkout() -> tuple[str, bool]:
    named = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR")
    return os.path.abspath(named or os.path.expanduser("~/ComfyUI")), named is not None


def test_the_staged_flux3_files_pass_and_the_fp8r_variant_refuses():
    directory = Path(_comfy_checkout()[0]) / "models" / "diffusion_models"
    fp8r = sorted(directory.glob("flux3_image_klein*fp8r*.safetensors"))
    if not fp8r or any(not (directory / name).is_file() for name in VENDOR_DITS):
        pytest.skip("the FLUX.3 vendor files are not staged in this ComfyUI checkout")
    for name in VENDOR_DITS:
        path = str(directory / name)
        assert upstream_gate.uncovered_scaled_layers(path) == (), name
        assert upstream_gate.refusal_for(path) is None, name
    for path in fp8r:
        layers = upstream_gate.uncovered_scaled_layers(str(path))
        assert len(layers) > 0
        found = upstream_gate.refusal_for(str(path))
        assert found is not None and found[1] == 120


# SenseNova's two top-level keys (stock_prefix_drift). The newest ComfyUI release
# keeps a file that holds both whole; master strips the prefix it counts.
SENSENOVA = {key: ("BF16", [8]) for key in upstream_gate._SENSENOVA_KEYS}
# The five names master's checkpoint conversion keys on, the rule's two first.
SENSENOVA_FILE = (*upstream_gate._SENSENOVA_KEYS, "fm_modules.fm_head.conv1.weight",
                  "language_model.model.embed_tokens.weight", "language_model.model.norm.weight")
NET6 = [f"net.blocks.{i}.weight" for i in range(6)]
UNDER_NET = {name: ("BF16", [8]) for name in NET6}
TOP_FP8 = {"fm_modules.extra.weight": ("F8_E4M3", [32, 32]),
           "fm_modules.extra.weight_scale": ("F32", [32])}
NET_FP8 = {"net.blocks.9.q.weight": ("F8_E4M3", [32, 32]), "net.blocks.9.q.weight_scale": ("F32", [32])}
# (tensors, metadata, refused under the release rule, refused under master's)
TOP_LEVEL_BACKBONES: dict[str, tuple[Spec, dict[str, str] | None, bool, bool]] = {
    "an_unrecorded_top_level_scale": ({**SENSENOVA, **UNDER_NET, **TOP_FP8}, None, True, False),
    "a_table_naming_the_stripped_layer": (
        {**SENSENOVA, **UNDER_NET, **NET_FP8}, _layers("blocks.9.q", fmt="float8_e4m3fn"), True, False),
    "an_unrecorded_scale_under_net": ({**SENSENOVA, **UNDER_NET, **NET_FP8}, None, True, True),
    # The published layout: every key at the top level, so both rules keep them all.
    "nothing_under_a_prefix": ({**SENSENOVA, **TOP_FP8}, None, True, True),
}
# (names, the prefix under the release rule, the prefix under master's)
TOP_LEVEL_PREFIXES = {
    "the_two_keys_alone": ([*SENSENOVA], "", "model."),
    "beside_six_under_net": ([*SENSENOVA, *NET6], "", "net."),
    "beside_five_under_net": ([*SENSENOVA, *NET6[:5]], "", "model."),
    "beside_six_under_the_ldm_prefix": (
        [*SENSENOVA, *(f"{PREFIX}blocks.{i}.weight" for i in range(6))], "", PREFIX),
    "beside_two_under_model": ([*SENSENOVA, "model.a.weight", "model.b.weight"], "", "model."),
    "one_key_beside_six_under_net": ([upstream_gate._SENSENOVA_KEYS[0], *NET6], "net.", "net."),
    "beside_detector_and_tracker": (["detector.a.weight", "tracker.b.weight", *SENSENOVA, *NET6], "", ""),
    # The five names master's checkpoint conversion keys on (comfy/checkpoint_convert.py),
    # so a rule moved to another pair of them turns the canary red.
    "a_whole_sensenova_file_beside_six_under_net": ([*SENSENOVA_FILE, *NET6], "", "net."),
}
TOP_SCALE = "an_unrecorded_top_level_scale"  # refused under the release rule alone


def _number(path: str, comfy_dir: Path | None = None) -> int | None:
    found = upstream_gate.refusal_for(path, comfy_dir=comfy_dir)
    return None if found is None else found[1]


@pytest.mark.parametrize("case", sorted(TOP_LEVEL_PREFIXES))
def test_the_replay_holds_both_comfy_prefix_rules(case):
    names, release, master = TOP_LEVEL_PREFIXES[case]
    keys = dict.fromkeys(names, "BF16")
    assert upstream_gate._unet_prefix(keys, sensenova_whole=True) == release
    assert upstream_gate._unet_prefix(keys, sensenova_whole=False) == master


def test_the_rule_is_read_from_the_checkout_and_nowhere_else(tmp_path):
    """A named checkout is read as text, from the last prefix function it defines, the
    one Python runs: both keys in a clause of its own that returns no prefix are the
    release rule, none of the file's keys is master's. A rule on one key, on another
    key of the file, on pieces of a key or on a name the module binds, a clause on
    both keys that returns another prefix, raises or tests more, and a source that
    is absent, compiled only or unparseable, cannot be told."""
    keeps = upstream_gate.keeps_sensenova_whole
    first, second = upstream_gate._SENSENOVA_KEYS
    one_key = MASTER.replace("    return", f"    if {first!r} in state_dict:\n        return ''\n    return", 1)
    cases = {
        "release": (RELEASE, True),
        "master": (MASTER, False),
        "the_keys_outside_the_function": (f"KEYS = {upstream_gate._SENSENOVA_KEYS!r}\n\n{MASTER}", False),
        "the_release_rule_defined_again_as_masters": (f"{RELEASE}\n\n{MASTER}", False),
        "masters_rule_defined_again_as_the_release_rule": (f"{MASTER}\n\n{RELEASE}", True),
        "one_key_alone": (one_key, None),
        "another_key_of_the_file": (RELEASE.replace(repr(second), repr(SENSENOVA_FILE[2])), None),
        "a_third_key_of_the_file": (RELEASE.replace("in state_dict:", f"in state_dict and {SENSENOVA_FILE[2]!r} "
                                                    "in state_dict:"), None),
        "a_key_in_pieces": (RELEASE.replace(repr(first), f"{first[:11]!r} + {first[11:]!r}"), None),
        "the_keys_in_a_constant_the_function_reads": (
            f"KEYS = {upstream_gate._SENSENOVA_KEYS!r}\n\n"
            + MASTER.replace("    return", "    if all(key in state_dict for key in KEYS):\n"
                             "        return ''\n    return", 1), None),
        "another_prefix_for_the_two_keys": (RELEASE.replace("        return ''\n", "        return 'fm_modules.'\n", 1),
                                            None),
        "a_clause_that_raises": (RELEASE.replace("        return ''\n", "        raise ValueError('convert it')\n", 1),
                                 None),
        "whole_only_without_a_strong_prefix": (RELEASE.replace(
            "in state_dict:\n", "in state_dict and not any(key.startswith('net.') for key in state_dict):\n", 1),
            None),
        "the_rule_in_a_helper": (
            f"def _whole(state_dict):\n    return {first!r} in state_dict and {second!r} in state_dict\n\n\n"
            + MASTER.replace("    return", "    if _whole(state_dict):\n        return ''\n    return", 1), None),
        "no_prefix_function": ("PREFIXES = ('net.',)\n", None),
        "half_written": (RELEASE[:60], None),
        "too_deep_to_parse": ("x = " + "1+" * 200000 + "1\n", None),
        "too_large_to_parse": ("x = " + "not " * 100000 + "1\n", None),
    }
    for name, (text, rule) in cases.items():
        assert keeps(write_checkout(tmp_path / name, text)) is rule, name
    assert keeps(str(write_checkout(tmp_path / "named_as_text", RELEASE))) is True
    assert keeps(tmp_path / "absent") is None
    assert keeps(compiled_only(tmp_path / "compiled", RELEASE)) is None


@pytest.mark.parametrize("case", sorted(TOP_LEVEL_BACKBONES))
def test_one_file_gets_each_checkouts_own_verdict(tmp_path, case):
    """One file and one process, three checkouts: the cached header read keys on the
    rule, and a checkout whose rule cannot be read refuses what either rule refuses."""
    tensors, metadata, release_refuses, master_refuses = TOP_LEVEL_BACKBONES[case]
    path = _write_header(tmp_path / f"{case}.safetensors", tensors, metadata)
    for name, text, refuses in (("release", RELEASE, release_refuses), ("master", MASTER, master_refuses),
                                ("unreadable", "def (\n", release_refuses or master_refuses)):
        assert _number(path, write_checkout(tmp_path / name, text)) == (120 if refuses else None), name


def test_a_rule_that_cannot_be_told_reads_a_file_twice_only_if_it_holds_both_keys(tmp_path, monkeypatch):
    """With no rule to read, a file is read under the second rule only when it holds
    both rule keys, the only files the two rules read differently. Any other file
    costs the one header read it always cost; the FSDP launch proof's count of
    header reads leans on that (tests/test_fsdp_capacity.py)."""
    from dgx_monarch import safetensors_header

    monkeypatch.delitem(sys.modules, "comfy.model_detection", raising=False)
    monkeypatch.setitem(sys.modules, "comfy", package())
    assert upstream_gate.keeps_sensenova_whole() is None
    reads: list[str] = []
    read = safetensors_header.read_safetensors_header
    monkeypatch.setattr(safetensors_header, "read_safetensors_header",
                        lambda path: reads.append(str(path)) or read(path))
    plain = _write_header(tmp_path / "plain.safetensors", *PASSES["bf16"])
    keyed = _write_header(tmp_path / "keyed.safetensors", {**SENSENOVA, **UNDER_NET})
    assert _number(plain) is None and _number(keyed) is None
    assert reads == [plain, keyed, keyed]


THIRD_ANSWER = "def unet_prefix_from_state_dict(state_dict):\n    return 'fm_modules.'\n"
BY_SHAPE = ("def unet_prefix_from_state_dict(state_dict):\n"
            "    return '' if state_dict[next(iter(state_dict))].shape else 'net.'\n")
# (the source the process loaded, what its file says on disk now, the folders ahead
# of it on comfy's namespace path, and the refusal number for the TOP_SCALE file)
LOADED = {
    "release": (RELEASE, RELEASE, [], 120),
    "master": (MASTER, MASTER, [], None),
    "disk_unchanged_and_a_custom_nodes_folder_first": (RELEASE, RELEASE, [MASTER], 120),
    "disk_changed_after_the_load": (RELEASE, MASTER, [], 120),
    "a_third_answer": (THIRD_ANSWER, THIRD_ANSWER, [], 120),
    "a_rule_that_reads_more_than_names": (BY_SHAPE, BY_SHAPE, [], 120),
}
# (comfy's namespace path as (source, compiled only), or None for no ComfyUI at all,
# and the refusal number for the TOP_SCALE file)
NOT_LOADED: dict[str, tuple[list[tuple[str, bool]] | None, int | None]] = {
    "release": ([(RELEASE, False)], 120),
    "master": ([(MASTER, False)], None),
    "masters_folder_first": ([(MASTER, False), (RELEASE, False)], None),
    "a_compiled_only_folder_first": ([(RELEASE, True), (MASTER, False)], 120),
    "no_comfy_to_import": (None, 120),
}


@pytest.mark.parametrize("state", sorted(LOADED))
def test_a_process_that_loaded_detection_asks_its_own_function(models, monkeypatch, tmp_path_factory, state):
    """Once ``comfy.model_detection`` is loaded its function answers, since that code
    runs: whatever its file on disk now says, and whichever folder now comes first on
    ``comfy``'s namespace path. A third answer, or a function that reads more than
    names, cannot be told, and the file is refused if either rule refuses it."""
    ran_text, disk_text, ahead, number = LOADED[state]
    name = "top_level.safetensors"
    _write_header(models / name, *TOP_LEVEL_BACKBONES[TOP_SCALE][:2])
    base = tmp_path_factory.mktemp("checkouts")
    ran = write_checkout(base / "ran", ran_text)
    module = loaded(ran)
    write_checkout(ran, disk_text)
    folders = [write_checkout(base / f"ahead{index}", text) for index, text in enumerate(ahead)]
    monkeypatch.setitem(sys.modules, "comfy", package(*folders, ran))
    monkeypatch.setitem(sys.modules, "comfy.model_detection", module)
    assert _number(str(models / name)) == number
    if number is None:
        loader_preflight.preflight_upstream_gated_artifact(name)
    else:
        with pytest.raises(UnsupportedModelError, match=r"#120\.$"):
            loader_preflight.preflight_upstream_gated_artifact(name)


@pytest.mark.parametrize("state", sorted(NOT_LOADED))
def test_a_process_that_has_not_loaded_detection_reads_what_it_would_import(
        models, monkeypatch, tmp_path_factory, state):
    """Before ``comfy.model_detection`` loads, as at a Worker's first load, the gate
    reads the source the process would import, in ``comfy``'s namespace path order.
    A compiled-only file there, or no ComfyUI to import at all, cannot be told."""
    sources, number = NOT_LOADED[state]
    name = "top_level.safetensors"
    _write_header(models / name, *TOP_LEVEL_BACKBONES[TOP_SCALE][:2])
    monkeypatch.delitem(sys.modules, "comfy.model_detection", raising=False)
    if sources is None:
        monkeypatch.delitem(sys.modules, "comfy", raising=False)
        monkeypatch.setattr(sys, "path", [entry for entry in sys.path if not (Path(entry) / "comfy").is_dir()])
        assert upstream_gate.keeps_sensenova_whole() is None
    else:
        base = tmp_path_factory.mktemp("checkouts")
        monkeypatch.setitem(sys.modules, "comfy", package(*(
            (compiled_only if compiled else write_checkout)(base / f"folder{index}", text)
            for index, (text, compiled) in enumerate(sources))))
    assert _number(str(models / name)) == number


def test_both_funnels_read_the_rule_of_the_comfy_they_imported(models, monkeypatch, tmp_path_factory):
    """The driver and a Worker name no checkout and read the one they imported
    ComfyUI from; a headless caller that names it gets the same answer."""
    name = "top_level.safetensors"
    _write_header(models / name, *TOP_LEVEL_BACKBONES[TOP_SCALE][:2])
    monkeypatch.delitem(sys.modules, "comfy.model_detection", raising=False)
    for with_rule in (True, False):
        checkout = write_checkout(tmp_path_factory.mktemp("checkout"), RELEASE if with_rule else MASTER)
        monkeypatch.setitem(sys.modules, "comfy", package(checkout))
        headless = upstream_gate.refusal_for(str(models / name), comfy_dir=checkout)
        worker = _worker()
        if with_rule:
            assert headless is not None and headless[1] == 120
            with pytest.raises(UnsupportedModelError, match=r"#120\.$"):
                loader_preflight.preflight_upstream_gated_artifact(name)
            with pytest.raises(UnsupportedModelError, match=r"#120\.$"):
                store_fsdp.ensure(worker, name, {}, [])
            assert worker.store.calls == []
        else:
            assert headless is None
            loader_preflight.preflight_upstream_gated_artifact(name)
            store_fsdp.ensure(worker, name, {}, [])
            assert len(worker.store.calls) == 1


# Real comfy, CPU only. Keep it last in the file: the conftest comfy guard
# checks the namespace once this module's fixtures are torn down.
def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def real_comfy():
    """Import real comfy.ops on CPU, then undo it (tests/test_auto_latent_downscale.py)."""
    comfy_dir, required = _comfy_checkout()
    preserved = {name: module for name, module in sys.modules.items() if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-dropped-scale-gate", "--cpu"]
        try:
            if not required:
                pytest.importorskip(
                    "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            import comfy.options

            comfy.options.enable_args_parsing()
            import comfy.memory_management
            import comfy.model_detection
            import comfy.ops
            import comfy.sd
            import comfy.utils
        finally:
            sys.argv = original_argv
        yield SimpleNamespace(ops=comfy.ops, utils=comfy.utils, sd=comfy.sd,
                              model_detection=comfy.model_detection,
                              memory_management=comfy.memory_management)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _comfy_loaded_keys(comfy, monkeypatch, tensors: Spec, metadata) -> set[str]:
    """The keys ComfyUI's own ``load_diffusion_model_state_dict`` hands detection.

    Detection is stubbed to capture its state dict and find no model, so the
    call returns None after the conversions and the prefix strip, which are
    ComfyUI's code. Values do not matter to the names, except that a
    ``scale_input`` of exactly 1.0 is dropped; every tensor here is zero.
    """
    import torch

    captured: dict[str, None] = {}

    def capture(sd, _prefix, metadata=None):
        captured.update(dict.fromkeys(sd))

    monkeypatch.setattr(comfy.model_detection, "model_config_from_unet", capture)
    monkeypatch.setattr(comfy.model_detection, "convert_diffusers_mmdit", lambda *_a: None)
    monkeypatch.setattr(comfy.model_detection, "model_config_from_diffusers_unet", lambda *_a: None)
    state = {key: torch.zeros(2 if key.endswith("scaled_fp8") else 1) for key in tensors}
    assert comfy.sd.load_diffusion_model_state_dict(state, metadata=metadata) is None
    assert captured, "detection never ran"
    return set(captured)


@pytest.mark.parametrize("case", sorted(PASSES) + sorted(REFUSALS))
def test_the_replay_names_the_keys_comfys_own_load_hands_the_model(real_comfy, monkeypatch, case):
    """The gate's names-only replay of comfy.sd.load_diffusion_model_state_dict
    equals ComfyUI's own, key for key, on every pass and refuse fixture."""
    tensors, metadata = {**PASSES, **REFUSALS}[case][:2]
    dtypes = {key: dtype for key, (dtype, _shape) in tensors.items()}
    replay = upstream_gate.comfy_load_keys(dtypes, metadata or {})
    assert set(replay) == _comfy_loaded_keys(real_comfy, monkeypatch, tensors, metadata)


PREFIX_CASES = {
    "bare_keys": ["single_blocks.0.q_proj.weight", "img_in.weight"],
    "five_under_net": [f"net.blocks.{i}.weight" for i in range(5)],
    "six_under_net": [f"net.blocks.{i}.weight" for i in range(6)],
    "a_tie_takes_the_first_candidate": [f"{prefix}blocks.{i}.weight" for i in range(6)
                                        for prefix in ("net.", "model.model.")],
    "a_key_counts_once": [f"model.diffusion_model.blocks.{i}.weight" for i in range(6)],
    "detector_and_tracker": ["detector.a.weight", "tracker.b.weight",
                             *(f"net.blocks.{i}.weight" for i in range(6))],
    "two_top_level_backbones": [*upstream_gate._SENSENOVA_KEYS,
                                *(f"net.blocks.{i}.weight" for i in range(6))],
}


@pytest.mark.parametrize("case", sorted(PREFIX_CASES))
def test_the_replay_strips_the_prefix_comfy_picks(real_comfy, case):
    keys = dict.fromkeys(PREFIX_CASES[case], "BF16")
    assert upstream_gate._unet_prefix(keys) == (
        real_comfy.model_detection.unet_prefix_from_state_dict(dict(keys)))


def _load_unrecorded(comfy, ops, aimdo: bool, layout: str = "U8"):
    """ComfyUI's own load of a quantized weight and its scale, no quant record.

    ``convert_old_quants`` then ``load_state_dict(strict=False)``, the calls
    ``BaseModel.load_model_weights`` makes, on the Linear ComfyUI builds for a
    file with no quantization record. ``U8`` is the nvfp4 layout (packed at
    half width, an F8 block scale); ``F8_E4M3`` is the fp8r layout and ``I8``
    an int8 one (a full-width weight of ones, an F32 per-row scale of 0.5).
    """
    import torch

    out_features, in_features = 8, 32
    if layout == "U8":
        state = {"lin.weight": torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
                 "lin.weight_scale": torch.zeros(out_features, in_features // 16,
                                                 dtype=torch.float8_e4m3fn)}
    else:
        dtype = {"F8_E4M3": torch.float8_e4m3fn, "I8": torch.int8}[layout]
        state = {"lin.weight": torch.ones(out_features, in_features, dtype=dtype),
                 "lin.weight_scale": torch.full((out_features,), 0.5)}
    saved = comfy.memory_management.aimdo_enabled
    comfy.memory_management.aimdo_enabled = aimdo
    try:
        module = torch.nn.Module()
        module.lin = ops.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
        converted, _metadata = comfy.utils.convert_old_quants(state, "", {})
        result = module.load_state_dict(converted, strict=False)
    finally:
        comfy.memory_management.aimdo_enabled = saved
    return module, result, in_features


@pytest.mark.parametrize("op_set", ["disable_weight_init", "manual_cast"])
def test_comfy_drops_a_packed_scale_with_no_record_under_dynamic_vram(real_comfy, tmp_path, op_set):
    """Why U8 is in the gate's dtype set.

    The legacy patcher's load raises on the halved width. DynamicVRAM's lazy
    load (``disable_weight_init._lazy_load_from_state_dict``) assigns the
    packed weight unchecked and lists the scale as unexpected, so nothing
    refuses until the first matmul. The gate refuses the same layout on disk.
    """
    import torch

    ops = getattr(real_comfy.ops, op_set)
    with pytest.raises(RuntimeError, match=r"size mismatch for lin\.weight"):
        _load_unrecorded(real_comfy, ops, aimdo=False)
    module, result, in_features = _load_unrecorded(real_comfy, ops, aimdo=True)
    assert list(result.unexpected_keys) == ["lin.weight_scale"]
    assert module.lin.weight.dtype == torch.uint8
    assert tuple(module.lin.weight.shape) == (8, in_features // 2)
    module.lin.comfy_cast_weights = True
    with pytest.raises(RuntimeError):
        module.lin(torch.ones(1, in_features, dtype=torch.bfloat16))
    path = _write_header(tmp_path / "packed.safetensors", {
        "lin.weight": ("U8", [8, in_features // 2]),
        "lin.weight_scale": ("F8_E4M3", [8, in_features // 16])})
    assert upstream_gate.uncovered_scaled_layers(path) == ("lin",)


@pytest.mark.parametrize("layout", ["F8_E4M3", "I8"], ids=["fp8r", "int8"])
@pytest.mark.parametrize("aimdo", [False, True], ids=["legacy", "dynamic_vram"])
def test_comfy_loads_an_unrecorded_layer_without_error_and_drops_its_scale(
        real_comfy, tmp_path, aimdo, layout):
    """The fp8r shape, and the same shape at int8: no error on either load
    path, the scale is unexpected, and the forward runs on the unscaled
    weight. This is what the gate refuses."""
    import torch

    ops = real_comfy.ops.manual_cast
    module, result, in_features = _load_unrecorded(real_comfy, ops, aimdo=aimdo, layout=layout)
    assert list(result.unexpected_keys) == ["lin.weight_scale"]
    module.lin.comfy_cast_weights = True
    out = module.lin(torch.ones(1, in_features, dtype=torch.bfloat16))
    assert torch.equal(out, torch.full_like(out, float(in_features)))  # scale 0.5 ignored
    path = _write_header(tmp_path / f"{layout}_layer.safetensors", {
        "lin.weight": (layout, [8, in_features]), "lin.weight_scale": ("F32", [8])})
    assert upstream_gate.uncovered_scaled_layers(path) == ("lin",)


def test_the_rule_the_gate_reads_from_the_checkout_is_comfys_own(real_comfy):
    """ComfyUI's own function, asked about the rule's two keys and about a whole
    SenseNova file, each beside six keys under net.: no prefix for both where the
    release rule holds, ``net.`` for both where master's does. Any other pair is a
    rule the replay does not hold. The gate's answer from the loaded function and
    its reading of the checkout's text must both be that rule."""
    rule = real_comfy.model_detection.unet_prefix_from_state_dict
    answers = [rule(dict.fromkeys([*names, *NET6])) for names in (SENSENOVA, SENSENOVA_FILE)]
    assert answers in (["", ""], ["net.", "net."]), (
        f"ComfyUI gives {answers!r} to the two keys and to a whole SenseNova file beside six "
        "under net.: a prefix rule stock_prefix_drift does not hold")
    assert upstream_gate.keeps_sensenova_whole() is (answers[0] == "")
    assert upstream_gate.keeps_sensenova_whole(_comfy_checkout()[0]) is (answers[0] == "")


@pytest.mark.parametrize("case", sorted(TOP_LEVEL_PREFIXES))
def test_the_replay_picks_comfys_prefix_beside_two_top_level_backbones(real_comfy, case):
    keys = dict.fromkeys(TOP_LEVEL_PREFIXES[case][0], "BF16")
    assert upstream_gate._unet_prefix(keys) == (
        real_comfy.model_detection.unet_prefix_from_state_dict(dict(keys)))


@pytest.mark.parametrize("case", sorted(TOP_LEVEL_BACKBONES))
def test_the_replay_names_comfys_keys_beside_two_top_level_backbones(real_comfy, monkeypatch, case):
    """The key-for-key check above, on files whose answer depends on the rule."""
    tensors, metadata = TOP_LEVEL_BACKBONES[case][:2]
    dtypes = {key: dtype for key, (dtype, _shape) in tensors.items()}
    replay = upstream_gate.comfy_load_keys(dtypes, metadata or {})
    assert set(replay) == _comfy_loaded_keys(real_comfy, monkeypatch, tensors, metadata)
