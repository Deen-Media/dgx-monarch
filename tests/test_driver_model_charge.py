"""nodes/driver_model_charge: what a driver node's model input costs, as the loader charges it.

LTX's audio-video text encoder loader names an LTX checkpoint for its text
projection alone, which ComfyUI keeps (comfy/text_encoders/lt.py) while it maps
and drops the rest; every other stock loader loads its model whole. The loader's
graph walk and the driver spend price's load wraps both read these figures, and
the text encoder that loader names is charged as one. CPU only, on small real
safetensors files.
"""
from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch import driver_spend_marks
from dgx_monarch.nodes import driver_model_charge, loader_graph

PROJECTION = {"text_embedding_projection.weight": (4, 4),
              "model.diffusion_model.video_embeddings_connector.proj.weight": (4, 8),
              "model.diffusion_model.audio_embeddings_connector.proj.weight": (2, 8)}
DIT = {"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": (64, 64)}


def _safetensors(path, tensors: dict) -> int:
    """A valid BF16 safetensors file of these tensors; returns its size."""
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = 2 * shape[0] * shape[1]
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path.stat().st_size


def _bytes(tensors: dict) -> int:
    return sum(2 * rows * cols for rows, cols in tensors.values())


@pytest.fixture
def files(monkeypatch, tmp_path):
    ltx = tmp_path / "ltx.safetensors"
    gemma = tmp_path / "gemma.safetensors"
    sizes = {"ltx.safetensors": _safetensors(ltx, {**PROJECTION, **DIT}),
             "gemma.safetensors": _safetensors(gemma, {"model.layers.0.w": (32, 32)})}
    paths = {"ltx.safetensors": str(ltx), "gemma.safetensors": str(gemma)}
    monkeypatch.setitem(sys.modules, "folder_paths",
                        SimpleNamespace(get_full_path=lambda folder, name: paths.get(name)))
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=True))
    monkeypatch.setitem(sys.modules, "comfy.utils", SimpleNamespace(DISABLE_MMAP=False))
    return SimpleNamespace(sizes=sizes, ltx=ltx)


def test_the_ltx_text_loader_charges_its_checkpoints_projection_alone(files):
    charge = driver_model_charge.driver_model_bytes(driver_model_charge.LTX_TEXT_LOADER, "ckpt_name",
                                                    "ltx.safetensors")
    assert charge == _bytes(PROJECTION) < files.sizes["ltx.safetensors"]


def test_every_other_loader_charges_the_whole_file(files):
    for class_type in ("CheckpointLoaderSimple", "ImageOnlyCheckpointLoader", ""):
        assert driver_model_charge.driver_model_bytes(class_type, "ckpt_name", "ltx.safetensors") == (
            files.sizes["ltx.safetensors"])


def test_a_copy_of_every_tensor_charges_the_whole_file(files, monkeypatch):
    """--disable-mmap with DynamicVRAM off: load_torch_file copies each tensor before load_clip drops it."""
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=False))
    monkeypatch.setitem(sys.modules, "comfy.utils", SimpleNamespace(DISABLE_MMAP=True))
    assert driver_model_charge.driver_model_bytes(driver_model_charge.LTX_TEXT_LOADER, "ckpt_name",
                                                  "ltx.safetensors") == files.sizes["ltx.safetensors"]


def test_an_unreadable_header_charges_the_whole_file(files):
    files.ltx.write_bytes(b"not a safetensors file at all")
    assert driver_model_charge.driver_model_bytes(driver_model_charge.LTX_TEXT_LOADER, "ckpt_name",
                                                  "ltx.safetensors") == files.ltx.stat().st_size


def test_the_loader_and_the_load_wrap_charge_the_ltx_text_loader_alike(files, monkeypatch):
    """The text encoder at the loader's ratio, the checkpoint's projection, and nothing more."""
    monkeypatch.setattr(loader_graph, "live_nodes", lambda graph: frozenset(graph))
    inputs = {"text_encoder": "gemma.safetensors", "ckpt_name": "ltx.safetensors", "device": "default"}
    expected = int(files.sizes["gemma.safetensors"] * loader_graph.TE_RESIDENT_RATIO) + _bytes(PROJECTION)
    assert driver_spend_marks.load_bytes(inputs, driver_model_charge.LTX_TEXT_LOADER) == expected
    graph = {"5": {"class_type": driver_model_charge.LTX_TEXT_LOADER, "inputs": inputs}}
    terms = loader_graph.driver_stack_terms(graph)
    assert terms.te_bytes == int(files.sizes["gemma.safetensors"] * loader_graph.TE_RESIDENT_RATIO)
    assert terms.model_bytes == _bytes(PROJECTION)
