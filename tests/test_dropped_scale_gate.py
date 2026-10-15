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
