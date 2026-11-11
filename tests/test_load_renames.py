"""actor/load_renames.py: the file key of a tensor ComfyUI renames as it loads a checkpoint.

A LoRA names the model's key; a bake that restores from the file needs the
file's. The inverse is taken from the model config's own
``process_unet_state_dict``, run over the header's names. The first tests use a
stand-in config; the last two run ComfyUI's real configs (skipped where
ComfyUI is not importable): FLUX's ``_norm.scale`` rule, the one the
2026-11-11 uly2+fsdp crash met, and every registered config, none of which may
raise through ``load_time_renames``.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.actor.load_renames import load_time_renames, renamed_file_key
from module_location_helpers import from_checkout


def _flux_rule(state_dict):
    return {(k[:-len(".scale")] + ".weight" if k.endswith("_norm.scale") else k): v
            for k, v in state_dict.items()}


def _model(process=_flux_rule):
    return SimpleNamespace(model_config=SimpleNamespace(process_unet_state_dict=process))


@pytest.mark.parametrize("namespace", ["", "diffusion_model.", "model.diffusion_model."])
def test_a_renamed_tensor_maps_back_to_its_file_key(namespace):
    header = {f"{namespace}double_blocks.0.img_attn.norm.query_norm.scale": object(),
              f"{namespace}double_blocks.0.img_attn.qkv.weight": object()}
    renames = load_time_renames(_model(), header)
    assert renames["double_blocks.0.img_attn.norm.query_norm.weight"] == (
        f"{namespace}double_blocks.0.img_attn.norm.query_norm.scale")
    # Only renamed tensors appear; the patcher's key finds it with or without its prefix.
    assert "double_blocks.0.img_attn.qkv.weight" not in renames
    for key in ("diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight",
                "double_blocks.0.img_attn.norm.query_norm.weight"):
        assert renamed_file_key(renames, key) == f"{namespace}double_blocks.0.img_attn.norm.query_norm.scale"
    assert renamed_file_key(renames, "diffusion_model.double_blocks.0.img_attn.qkv.weight") is None


def test_no_config_no_function_or_a_function_that_reads_tensors_yields_nothing():
    header = {"a_norm.scale": object()}
    assert load_time_renames(SimpleNamespace(), header) == {}
    assert load_time_renames(SimpleNamespace(model_config=SimpleNamespace()), header) == {}
    assert load_time_renames(_model(lambda sd: {k: v * 2.5 for k, v in sd.items()}), header) == {}
    assert load_time_renames(_model(lambda sd: None), header) == {}


def test_a_value_that_is_not_a_header_name_is_dropped():
    """A config that builds new tensors (not a rename) names nothing in the file."""
    header = {"a.weight": object(), "b.weight": object()}
    merged = _model(lambda sd: {"ab.weight": sd["a.weight"] + sd["b.weight"], "a.weight": sd["a.weight"]})
    assert load_time_renames(merged, header) == {}


@pytest.fixture(scope="module")
def supported_models():
    """ComfyUI's real model configs with --cpu, restored afterwards."""
    def is_comfy(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved = {name: module for name, module in sys.modules.items() if is_comfy(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-load-renames", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
            options.enable_args_parsing()
            module = pytest.importorskip("comfy.supported_models")
        finally:
            sys.argv = argv_before
        yield module
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _config(cls):
    try:
        return cls({})
    except Exception:
        return cls.__new__(cls)


def test_comfyuis_flux_configs_map_a_norm_weight_to_its_scale(supported_models):
    """The crash's own key, through ComfyUI's own rename, for FLUX and FLUX.2."""
    file_key = "double_blocks.0.img_attn.norm.query_norm.scale"
    header = {file_key: object(), "double_blocks.0.img_attn.qkv.weight": object(),
              "single_blocks.8.norm.key_norm.scale": object()}
    for name in ("Flux", "Flux2"):
        model = SimpleNamespace(model_config=_config(getattr(supported_models, name)))
        renames = load_time_renames(model, header)
        assert renamed_file_key(
            renames, "diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight") == file_key, name
        assert renamed_file_key(
            renames, "diffusion_model.single_blocks.8.norm.key_norm.weight") == "single_blocks.8.norm.key_norm.scale"
        assert renamed_file_key(renames, "diffusion_model.double_blocks.0.img_attn.qkv.weight") is None


def test_no_registered_config_raises_and_every_rename_names_a_header_key(supported_models):
    header = {name: object() for name in (
        "double_blocks.0.img_attn.norm.query_norm.scale", "blocks.0.norm.scale", "a.b.scale",
        "double_blocks.0.img_attn_qkv.weight", "single_blocks.0.modulation.linear.weight",
        "txt_in.t_embedder.mlp.0.weight", "blocks.0.attn.qkv.weight", "final_layer.linear.weight")}
    renaming = []
    for cls in supported_models.models:
        renames = load_time_renames(SimpleNamespace(model_config=_config(cls)), header)
        assert all(value in header for value in renames.values()), cls.__name__
        if renames:
            renaming.append(cls.__name__)
    assert "Flux2" in renaming and "Flux" in renaming, renaming
