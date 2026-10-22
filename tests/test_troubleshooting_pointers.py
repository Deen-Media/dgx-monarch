"""Every refusal pointer names a docs/TROUBLESHOOTING.md entry that exists.

A refusal ends "See docs/TROUBLESHOOTING.md #N." from ``refusal(...,
troubleshooting=N)``, and some texts name the entry inline. Entries are
numbered in order, so a merge that renumbers them leaves a pointer on a
neighboring entry with the suite green unless something reads the headings.
The first test reads every pointer in src. The second raises the MOSS-Image,
Skip Layer Guidance, LoRA quantization and reference-image refusals and checks
that each lands on its own entry, and the third checks the documentation links
to those entries.
"""
from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import lora_quant_guard
from dgx_monarch.actor import guidance_render
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.moss_image_seams import refuse_control_channels, refuse_reference_images
from dgx_monarch.nodes import model_patch_preflight as preflight
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.model_sampling import DGXMonarchModelSamplingSD3
from dgx_monarch.nodes.skip_layer_guidance import DGXMonarchSkipLayerGuidance
from dgx_monarch.topology import topology_from_preset

REPO = Path(__file__).resolve().parents[1]
TROUBLESHOOTING = REPO / "docs" / "TROUBLESHOOTING.md"
POINTERS = (re.compile(r"troubleshooting\s*=\s*(\d+)"), re.compile(r"TROUBLESHOOTING\.md\s*#(\d+)"))
SKIP = "Skip Layer Guidance (DGX Monarch) refuses on cfg2"
SHAPE = "A LoRA that changes a weight's shape refuses before any bake"
SHIFT = "Model Sampling SD3 (DGX Monarch) refuses on a family whose ComfyUI sampling multiplier is 1.0"
FLOOR = "A MOSS-Image checkpoint fails to load on ComfyUI older than 0.42.0"
LORA_QUANT = "A MOSS-Image INT8 render with a LoRA refuses on every topology"
REFERENCES = "A MOSS-Image render with reference images refuses on every topology but one GPU"


def _headings() -> dict[int, str]:
    return {int(number): title for number, title in
            re.findall(r"^## (\d+)\. (.+)$", TROUBLESHOOTING.read_text(), re.MULTILINE)}


def _slug(number: int, title: str) -> str:
    """GitHub's anchor for a ``## N. title`` heading."""
    text = re.sub(r"[^\w\- ]", "", f"{number}. {title}".lower())
    return text.replace(" ", "-")


def _pointed(text: str) -> int:
    found = {int(n) for n in re.findall(r"docs/TROUBLESHOOTING\.md #(\d+)", text)}
    assert len(found) == 1, text
    return found.pop()


def test_every_pointer_in_src_names_an_existing_entry():
    headings = _headings()
    seen = 0
    for path in sorted((REPO / "src").rglob("*.py")):
        text = path.read_text()
        for pattern in POINTERS:
            for match in pattern.finditer(text):
                seen += 1
                number = int(match.group(1))
                line = text.count("\n", 0, match.start()) + 1
                assert number in headings, (
                    f"{path.relative_to(REPO)}:{line} points at TROUBLESHOOTING #{number}, which has no heading")
    assert seen > 100, "the scan found almost no pointers; the patterns no longer match the source"
    assert sorted(headings) == list(range(1, max(headings) + 1)), "TROUBLESHOOTING numbers skip or repeat"


def _model(**kwargs) -> ModelSpec:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    return ModelSpec(mesh=mesh, unet_name="model.safetensors", options={"weight_dtype": "default"}, **kwargs)


def _skipping(start: float = 0.0) -> ModelSpec:
    patched, = DGXMonarchSkipLayerGuidance().patch(
        _model(), double_layers="", single_layers="22", start_percent=start, end_percent=1.0)
    return patched


def _driver_skip() -> None:
    preflight.preflight_skip_layer_guidance(_skipping(), topology_from_preset("cfg2", 2), {})


def _worker_skip() -> None:
    worker = SimpleNamespace(topology={"ulysses": 1, "ring": 1, "cfg": 2, "dp": 1, "fsdp": False})
    adapter = SimpleNamespace(family="moss_image", sp_honours_block_replace=True)
    guidance_render.apply_skip_layer_guidance(
        object(), {"model": _skipping().request_dict()}, worker, adapter)


def _shift_after_skip() -> None:
    DGXMonarchModelSamplingSD3().patch(_skipping(start=0.2), 3.0)


def _control_channels() -> None:
    refuse_control_channels(384, 128)


def _lora_quant() -> None:
    lora_quant_guard.assert_lora_quant_supported("moss_image", "int8", [{"name": "a.safetensors"}])


def _worker_references() -> None:
    refuse_reference_images([object()], None)


def _driver_references(monkeypatch) -> None:
    stub = types.ModuleType("folder_paths")
    stub.get_full_path = lambda _kind, _name: "/home/user/models/diffusion_models/m.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", stub)
    monkeypatch.setattr("dgx_monarch.adapters.detect.sniff_checkpoint",
                        lambda _path: ("moss_image", "bf16"))
    preflight.preflight_moss_image_reference_images(_model(), (True, None),
                                                    topology_from_preset("uly2", 2))


def _multiplier_one(monkeypatch) -> None:
    stub = types.ModuleType("folder_paths")
    stub.get_full_path = lambda _kind, _name: "/home/user/models/diffusion_models/m.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", stub)
    config = type("MOSSImage", (), {"sampling_settings": {"multiplier": 1.0, "shift": 5.0}})()
    monkeypatch.setattr("dgx_monarch.render_memory_price.detect_comfy_model_config", lambda _path: config)
    preflight.preflight_shift_node(_model().with_model_sampling({"kind": "sd3", "shift": 5.0}))


@pytest.mark.parametrize(("raise_it", "title"), [
    (_driver_skip, SKIP),
    (_worker_skip, SKIP),
    (_shift_after_skip, SKIP),
    (_control_channels, SHAPE),
    (_multiplier_one, SHIFT),
    (_lora_quant, LORA_QUANT),
    (_worker_references, REFERENCES),
    (_driver_references, REFERENCES),
], ids=["driver-skip", "worker-skip", "shift-after-skip", "control-channels", "multiplier-one",
        "lora-quant", "worker-references", "driver-references"])
def test_the_moss_and_carrier_refusals_land_on_their_own_entries(monkeypatch, raise_it, title):
    with pytest.raises(UnsupportedModelError) as caught:
        if raise_it in (_multiplier_one, _driver_references):
            raise_it(monkeypatch)
        else:
            raise_it()
    assert _headings()[_pointed(str(caught.value))].startswith(title)


def test_the_lora_shape_guard_points_at_its_entry():
    """The guard needs ComfyUI to raise, so this reads its refusal's pointer from the source."""
    tree = ast.parse((REPO / "src/dgx_monarch/actor/lora_shape_guard.py").read_text())
    numbers = {keyword.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
               for keyword in node.keywords
               if keyword.arg == "troubleshooting" and isinstance(keyword.value, ast.Constant)}
    assert len(numbers) == 1
    assert _headings()[numbers.pop()] == SHAPE


def test_the_documentation_links_to_the_moss_entries_name_their_headings():
    headings = _headings()
    titles = (FLOOR, SKIP, SHAPE, SHIFT, LORA_QUANT, REFERENCES)
    entries = {number: title for number, title in headings.items() if title.startswith(titles)}
    assert sorted(title[:24] for title in entries.values()) == sorted(title[:24] for title in titles)
    anchor = re.compile(r"(?:TROUBLESHOOTING\.md)?#(\d+)-([\w\-]+)\)")
    linked = 0
    for path in [REPO / "CHANGELOG.md", *sorted((REPO / "docs").glob("*.md"))]:
        for number, rest in anchor.findall(path.read_text()):
            if int(number) in entries:
                linked += 1
                assert f"{number}-{rest}" == _slug(int(number), entries[int(number)]), (
                    f"{path.name} links #{number}-{rest}, not the heading's anchor")
    install = (REPO / "docs" / "INSTALL.md").read_text()
    floor = next(number for number, title in entries.items() if title == FLOOR)
    assert f"TROUBLESHOOTING.md#{_slug(floor, FLOOR)})" in install
    assert linked >= 12
