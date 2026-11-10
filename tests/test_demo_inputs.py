"""Every shipped template's demo inputs ship with the pack and reach ComfyUI's input folder.

The final hardware pass of the 1.5 round found the Qwen Image 2.1 masked edit
template naming qwen21_source.png, a file no install had: the pack shipped no
demo input, and ComfyUI ships only input/example.png. The pack now ships every
name in example_workflows/input (tools/gen_demo_inputs.py) and copies the ones
the input folder lacks at import (src/dgx_monarch/demo_inputs.py). No test here
touches a real ComfyUI input folder.
"""
from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from dgx_monarch import demo_inputs

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "example_workflows").glob("*.json"))
SHIPPED = ROOT / "example_workflows" / "input"
COMFY_OWN = {"example.png"}  # a tracked file of every ComfyUI checkout
MEDIA = {"LoadImage": 0, "LoadImageMask": 0, "LoadVideo": 0, "LoadAudio": 0}
# The frames a template takes from each clip: WanSCAILToVideo and
# WanAnimate2ToVideo read 81, FlowRVS's Image From Batch reads 17.
MIN_FRAMES = {"driving_video.mp4": 81, "driving_video_mask.mp4": 81,
              "wan_animate2_driving.mp4": 81, "source_video.mp4": 17}


@pytest.fixture(autouse=True)
def _switch_unset(monkeypatch):
    monkeypatch.delenv("DGXM_DEMO_INPUTS", raising=False)


def _nodes(graph: dict) -> list[dict]:
    found = list(graph.get("nodes", []))
    for subgraph in graph.get("definitions", {}).get("subgraphs", []):
        found.extend(_nodes(subgraph))
    return found


def _named() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for path in TEMPLATES:
        for node in _nodes(json.loads(path.read_text(encoding="utf-8"))):
            slot = MEDIA.get(node.get("type", ""))
            values = node.get("widgets_values")
            if slot is not None and isinstance(values, list) and isinstance(values[slot], str):
                names.setdefault(values[slot], set()).add(path.stem)
    return names


def test_every_demo_input_a_shipped_template_names_ships_with_the_pack():
    named = _named()
    assert len(named) > 20, "the scan found almost no input nodes; the widget layout moved"
    missing = {name: sorted(used) for name, used in named.items()
               if name not in COMFY_OWN and not (SHIPPED / name).is_file()}
    assert not missing, f"templates name demo inputs the pack does not ship: {missing}"


def test_the_pack_ships_only_names_a_template_uses_and_the_generator_writes_them_all():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import gen_demo_inputs
    finally:
        sys.path.remove(str(ROOT / "tools"))
    on_disk = {path.name for path in demo_inputs.shipped()}
    assert on_disk == set(_named()) - COMFY_OWN
    assert set(gen_demo_inputs.FILES) == on_disk


@pytest.mark.parametrize("name", sorted(path.name for path in SHIPPED.glob("*.png")))
def test_each_demo_image_opens_with_an_alpha_channel_where_its_template_reads_one(name):
    with Image.open(SHIPPED / name) as image:
        image.load()
        wants_alpha = name.endswith("_rgba.png") or "alpha" in name
        assert image.mode == ("RGBA" if wants_alpha else "RGB")
        assert min(image.size) >= 480
        if name in {"qwen21_mask_rgba.png", "qwen21_edit_mask_alpha.png"}:
            low, high = image.getchannel("A").getextrema()
            assert (low, high) == (0, 255), "the change guide needs a transparent region"


@pytest.mark.parametrize("name", sorted(MIN_FRAMES))
def test_each_demo_clip_decodes_with_the_frames_its_template_takes(name):
    av = pytest.importorskip("av", reason="PyAV, ComfyUI's video and audio reader, is absent here")
    with av.open(str(SHIPPED / name)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames >= MIN_FRAMES[name]


def test_the_demo_track_runs_about_as_long_as_the_wandancer_note_asks():
    av = pytest.importorskip("av", reason="PyAV, ComfyUI's video and audio reader, is absent here")
    with av.open(str(SHIPPED / "wan_dancer_reference_audio.mp3")) as container:
        stream = container.streams.audio[0]
        seconds = sum(frame.samples for frame in container.decode(audio=0)) / stream.rate
    assert 20.0 <= seconds <= 30.0


def test_placement_copies_each_absent_name_and_leaves_every_present_one(tmp_path):
    names = [path.name for path in demo_inputs.shipped()]
    kept, linked = tmp_path / names[0], tmp_path / names[1]
    kept.write_bytes(b"the owner's own file")
    linked.symlink_to(tmp_path / "nowhere.png")  # a dangling link still counts as present

    placed = demo_inputs.place_demo_inputs(tmp_path)

    assert placed == names[2:]
    assert kept.read_bytes() == b"the owner's own file"
    assert linked.is_symlink() and not linked.exists()
    for name in placed:
        assert (tmp_path / name).read_bytes() == (SHIPPED / name).read_bytes()
    assert demo_inputs.place_demo_inputs(tmp_path) == []


def test_placement_reads_comfys_input_folder_and_does_nothing_outside_comfy(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "folder_paths", None)  # import folder_paths now raises ImportError
    assert demo_inputs.place_demo_inputs() == []
    fake = types.ModuleType("folder_paths")
    fake.get_input_directory = lambda: str(tmp_path / "input")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    assert demo_inputs.place_demo_inputs() == [path.name for path in demo_inputs.shipped()]


@pytest.mark.parametrize("value", ["off", "0", "false", "No"])
def test_the_switch_turns_placement_off(monkeypatch, tmp_path, value):
    monkeypatch.setenv("DGXM_DEMO_INPUTS", value)
    assert demo_inputs.place_demo_inputs(tmp_path) == []
    assert not any(tmp_path.iterdir())


def test_a_folder_it_cannot_write_logs_a_warning_and_raises_nothing(monkeypatch, tmp_path):
    blocked = tmp_path / "input"
    blocked.write_text("a file where the folder should be")
    warnings: list[str] = []
    monkeypatch.setattr(demo_inputs.log, "warning", lambda message, *args: warnings.append(message % args))
    assert demo_inputs.place_demo_inputs(blocked) == []
    assert len(warnings) == 1 and "supply your own" in warnings[0]


def test_the_pack_entrypoint_places_the_demo_inputs_at_import():
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "place_demo_inputs"]
    assert len(calls) == 1 and not calls[0].args
