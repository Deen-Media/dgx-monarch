"""nodes/driver_model_charge: what a driver node's model input costs, as the loader charges it.

LTX's audio-video text encoder loader names an LTX checkpoint for its text
projection alone, which ComfyUI keeps (comfy/text_encoders/lt.py, ``load_sd``)
while it maps the file and reads nothing else of it under DynamicVRAM; every
other stock loader loads its model whole. The loader's graph walk and the driver
spend price's load wraps both read these figures, and the text encoder that
loader names is charged as one. CPU only: small real safetensors files, and the
real headers of this box's LTX checkpoints where they are on disk.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import driver_spend_marks, safetensors_header
from dgx_monarch.nodes import driver_model_charge, loader_graph

GIB = 2 ** 30
PROJECTION = {"text_embedding_projection.weight": (4, 4), "text_embedding_projection.bias": (4,)}
CONNECTOR = "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.{}.attn1.to_q.bias"
VIDEO_CONNECTOR = {"model.diffusion_model.video_embeddings_connector.proj.weight": (4, 8)}
DIT = {"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": (64, 64)}


def _safetensors(path, tensors: dict) -> int:
    """A valid BF16 safetensors file of these tensors; returns its size."""
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = 2 * math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path.stat().st_size


def _bytes(tensors: dict) -> int:
    return sum(2 * math.prod(shape) for shape in tensors.values())


def _connectors(compat: bool) -> dict:
    """An audio connector in compat mode (no third block, a 3840 wide first bias), or not."""
    if compat:
        return {**VIDEO_CONNECTOR, CONNECTOR.format(0): (3840,), CONNECTOR.format(1): (3840,)}
    return {**VIDEO_CONNECTOR, CONNECTOR.format(0): (2048,), CONNECTOR.format(2): (2048,)}


@pytest.fixture
def files(monkeypatch, tmp_path):
    paths: dict = {}

    def make(name: str, tensors: dict) -> int:
        paths[name] = str(tmp_path / name)
        return _safetensors(tmp_path / name, tensors)

    monkeypatch.setitem(sys.modules, "folder_paths",
                        SimpleNamespace(get_full_path=lambda folder, name: paths.get(name)))
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=True))
    monkeypatch.setattr(driver_model_charge, "_KEPT", {})
    return SimpleNamespace(make=make, paths=paths)


def _ltx(name: str) -> int:
    return driver_model_charge.driver_model_bytes(driver_model_charge.LTX_TEXT_LOADER, "ckpt_name", name)


def test_the_ltx_text_loader_charges_its_checkpoints_projection_alone(files):
    size = files.make("ltx.safetensors", {**PROJECTION, **_connectors(compat=False), **DIT})
    assert _ltx("ltx.safetensors") == _bytes(PROJECTION) < size


def test_in_compat_mode_the_connectors_are_kept_too(files):
    connectors = _connectors(compat=True)
    files.make("old.safetensors", {**PROJECTION, **connectors, **DIT})
    assert _ltx("old.safetensors") == _bytes(PROJECTION) + _bytes(connectors)


def test_every_other_loader_charges_the_whole_file(files):
    size = files.make("ltx.safetensors", {**PROJECTION, **DIT})
    for class_type in ("CheckpointLoaderSimple", "ImageOnlyCheckpointLoader", ""):
        assert driver_model_charge.driver_model_bytes(class_type, "ckpt_name", "ltx.safetensors") == size


def test_without_dynamic_vram_the_whole_file_is_charged(files, monkeypatch):
    """Without DynamicVRAM the safetensors library decides what a tensor holds; ComfyUI's code does not show it."""
    size = files.make("ltx.safetensors", {**PROJECTION, **DIT})
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=False))
    assert _ltx("ltx.safetensors") == size


def test_an_unreadable_header_charges_the_whole_file(files, tmp_path):
    files.make("ltx.safetensors", {**PROJECTION, **DIT})
    (tmp_path / "ltx.safetensors").write_bytes(b"not a safetensors file at all")
    assert _ltx("ltx.safetensors") == len(b"not a safetensors file at all")


def test_the_header_is_read_once_per_path_until_the_file_changes(files, monkeypatch, tmp_path):
    files.make("ltx.safetensors", {**PROJECTION, **DIT})
    reads: list = []
    read = safetensors_header.read_safetensors_header
    monkeypatch.setattr(safetensors_header, "read_safetensors_header", lambda path: reads.append(path) or read(path))
    assert _ltx("ltx.safetensors") == _ltx("ltx.safetensors") == _bytes(PROJECTION)
    assert len(reads) == 1
    files.make("ltx.safetensors", {**PROJECTION, **PROJECTION, "text_embedding_projection.extra": (8, 8), **DIT})
    stat = (tmp_path / "ltx.safetensors").stat()
    os.utime(tmp_path / "ltx.safetensors", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    assert _ltx("ltx.safetensors") == _bytes(PROJECTION) + 2 * 64 and len(reads) == 2


def test_the_loader_and_the_load_wrap_charge_the_ltx_text_loader_alike(files, monkeypatch):
    """The text encoder at the loader's ratio, the checkpoint's projection, and nothing more."""
    monkeypatch.setattr(loader_graph, "live_nodes", lambda graph: frozenset(graph))
    gemma = files.make("gemma.safetensors", {"model.layers.0.w": (32, 32)})
    files.make("ltx.safetensors", {**PROJECTION, **_connectors(compat=False), **DIT})
    inputs = {"text_encoder": "gemma.safetensors", "ckpt_name": "ltx.safetensors", "device": "default"}
    expected = int(gemma * loader_graph.TE_RESIDENT_RATIO) + _bytes(PROJECTION)
    assert driver_spend_marks.load_bytes(inputs, driver_model_charge.LTX_TEXT_LOADER) == expected
    terms = loader_graph.driver_stack_terms({"5": {"class_type": driver_model_charge.LTX_TEXT_LOADER,
                                                   "inputs": inputs}})
    assert terms.te_bytes == int(gemma * loader_graph.TE_RESIDENT_RATIO)
    assert terms.model_bytes == _bytes(PROJECTION)


def _box_file(folder: str, name: str) -> Path:
    path = Path(os.environ.get("COMFY_DIR", "")) / "models" / folder / name
    if not os.environ.get("COMFY_DIR") or not path.is_file():
        pytest.skip(f"{name} is not on this box")
    return path


@pytest.mark.parametrize("name", ["ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled-fp8.safetensors",
                                  "ltx-2.3-22b-dev-fp8.safetensors"])
def test_this_boxs_ltx_checkpoints_charge_their_projection(name, monkeypatch):
    """On the real headers: LTX 2.3 has the third audio block, so no compat mode, and only the projection,
    about 2.2 GiB, of a 27 to 43 GiB checkpoint is charged."""
    path = _box_file("checkpoints", name)
    monkeypatch.setattr(driver_model_charge, "_KEPT", {})
    kept = driver_model_charge.ltx_text_bytes(str(path))
    tensors = safetensors_header.read_safetensors_header(str(path)).tensors
    assert driver_model_charge._AUDIO_BLOCK.format(2) in tensors
    assert kept == sum(t.end - t.begin for key, t in tensors.items() if key.startswith("text_embedding_projection."))
    assert 1 * GIB < kept < 4 * GIB < path.stat().st_size


@pytest.mark.parametrize("name", ["gemma_3_12B_it.safetensors", "gemma_3_12B_it_fp4_mixed.safetensors"])
def test_this_boxs_gemma_files_are_charged_as_text_encoders(name, monkeypatch):
    path = _box_file("text_encoders", name)
    monkeypatch.setitem(sys.modules, "folder_paths",
                        SimpleNamespace(get_full_path=lambda folder, file: str(path) if file == name else None))
    assert driver_spend_marks.load_bytes({"text_encoder": name}) == int(
        path.stat().st_size * loader_graph.TE_RESIDENT_RATIO)
