"""Real ComfyUI: the MiniMax-H3-Image row rests on the facts ComfyUI's own detection reads.

ComfyUI 0.45.0 (upstream pull request 17655) names a MiniMax H3 file the image
model when ``audio_patch_proj.weight`` is absent, by ``video_patch_proj.weight``
and ``condition_proj.weight`` (``comfy/model_detection.py``), and builds
``MiniMaxH3ImageModel`` with ``pdd_sigmas`` from the file's ``config`` metadata
and no default for it. ``detect._SIGNATURES`` carries the same two paths after
H3's row, and the upstream gate refuses a file without the grid (#149). A
ComfyUI before 0.45.0 detects no image file at all, and never as H3 video.

These fail the day ComfyUI changes how it names the family: its needles, its
``image_model`` string, its config class or the grid's missing default. They run
on meta tensors built from ``minimax_h3_image_headers`` (the published files'
names, dtypes and shapes) and need no weights. Skipped where ComfyUI is not
importable; the comfy-canary job runs them against master and the newest release.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import minimax_h3_image_headers as h3i
from dgx_monarch.adapters import minimax_h3_image_header as h3_image
from dgx_monarch.adapters.detect import detect_family_from_keys
from module_location_helpers import from_checkout

_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16,
           "U8": torch.uint8, "I8": torch.int8, "F8_E4M3": torch.float8_e4m3fn}


def _comfy_dir() -> str:
    value = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR") or "~/ComfyUI"
    return os.path.abspath(os.path.expanduser(value))


# ComfyUI 0.45.0 added MiniMax-H3-Image; a tree at or past it must carry the family.
_FAMILY_SINCE = (0, 45, 0)


def _comfy_version(comfy_dir: str) -> tuple[int, ...] | None:
    """The checkout's ``comfyui_version.py`` version, read as tools/fp8_paths.py reads it."""
    path = Path(comfy_dir) / "comfyui_version.py"
    found = re.search(r'__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)', path.read_text()) if path.is_file() else None
    return tuple(int(part) for part in found.groups()) if found else None


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real comfy on its CPU path, then put sys.modules back as found."""
    comfy_dir = _comfy_dir()
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-minimax-h3-image-detection", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI checkout on sys.path")
            options.enable_args_parsing()
            detection = pytest.importorskip("comfy.model_detection")
            supported = pytest.importorskip("comfy.supported_models")
            minimax = pytest.importorskip("comfy.ldm.minimax.model")
        finally:
            sys.argv = original_argv
        version = _comfy_version(comfy_dir)
        # The no-family branch is allowed only on a ComfyUI known to predate the
        # family; an unknown version must carry it, so removed support fails here.
        older = version is not None and version < _FAMILY_SINCE
        has_family = hasattr(supported, "MiniMaxH3Image")
        assert has_family or older, (
            f"ComfyUI {version} at {comfy_dir} has no MiniMaxH3Image, which 0.45.0 added")
        yield SimpleNamespace(detection=detection, supported=supported, minimax=minimax,
                              has_family=has_family, older=older)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _state(kind: str, *, audio: bool = False) -> dict:
    return {name: torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
            for name, (dtype, shape, *_payload) in h3i.tensors(kind, blocks=2, audio=audio).items()}


@pytest.mark.parametrize("kind", h3i.KINDS)
def test_comfy_and_the_detector_name_the_same_family_from_the_same_keys(comfy, kind):
    image, video = _state(kind), _state(kind, audio=True)
    assert detect_family_from_keys(list(image)) == h3_image.FAMILY
    assert detect_family_from_keys(list(video)) == "minimax_h3"
    # The audio path names H3 video on every ComfyUI that knows H3.
    found = comfy.detection.detect_unet_config(video, "", metadata=h3i.METADATA)
    assert found is not None and found.get("image_model") == "minimax_h3"
    unet = comfy.detection.detect_unet_config(image, "", metadata=h3i.METADATA)
    if comfy.older:
        # Before 0.45.0 an image file detects as nothing, never as H3 video.
        assert unet is None
        return
    assert unet is not None, "this ComfyUI no longer detects MiniMax-H3-Image"
    assert unet["image_model"] == h3_image.FAMILY
    assert unet["pdd_sigmas"] == h3i.GRID and unet["pdd_steps"] == 6
    assert unet["latents_dim"] == 24  # 576 head rows over 4 * pdd_steps
    config = comfy.detection.model_config_from_unet(image, "", metadata=h3i.METADATA)
    assert type(config).__name__ == "MiniMaxH3Image"
    assert config.unet_config.get("image_model") == h3_image.FAMILY


def test_the_image_model_takes_no_default_for_its_grid(comfy):
    """Why the gate refuses a file without the grid: stock fails to build it."""
    if comfy.older:
        pytest.skip("this ComfyUI predates MiniMax-H3-Image (ComfyUI 0.45.0 added it)")
    grid = inspect.signature(comfy.minimax.MiniMaxH3ImageModel.__init__).parameters["pdd_sigmas"]
    assert grid.default is inspect.Parameter.empty
    image = _state("bf16")
    for metadata in (None, {"config": json.dumps({"transformer": {"pdd_steps": 6}})}):
        unet = comfy.detection.detect_unet_config(image, "", metadata=metadata)
        assert unet["image_model"] == h3_image.FAMILY and "pdd_sigmas" not in unet
        assert h3_image.grid_missing(metadata)
    assert not h3_image.grid_missing(h3i.METADATA)


def test_a_forced_minimax_h3_refuses_the_image_class_comfy_builds(comfy):
    """A forced ``family_adapter=minimax_h3`` binds by class ancestry, and ComfyUI's
    ``MiniMaxH3Image`` subclasses ``MiniMaxH3``. The video adapter must not take it."""
    if comfy.older:
        pytest.skip("this ComfyUI predates MiniMax-H3-Image (ComfyUI 0.45.0 added it)")
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    model_base = sys.modules["comfy.model_base"]
    with pytest.raises(UnsupportedModelError) as raised:
        adapter_for(object.__new__(model_base.MiniMaxH3Image), "minimax_h3")
    assert "MiniMaxH3Image" in str(raised.value)
    tag = parse_refusal_tag(str(raised.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert adapter_for(object.__new__(model_base.MiniMaxH3), "minimax_h3").family == "minimax_h3"
