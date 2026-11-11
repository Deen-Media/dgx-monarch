"""Every shipped template's demo inputs ship with the pack and reach ComfyUI's input folder.

The final hardware pass of the 1.5 round found the Qwen Image 2.1 masked edit
template naming qwen21_source.png, a file no install had: the pack shipped no
demo input, and ComfyUI ships only input/example.png. The pack now ships every
name in example_workflows/input (tools/gen_demo_inputs.py) and copies the ones
the input folder lacks at import (src/dgx_monarch/demo_inputs.py). No test here
touches a real ComfyUI input folder: the battery turns the copy off for every
test (tests/conftest.py), and these tests clear that switch only to point the
copy at scratch folders.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import types
from fnmatch import fnmatch
from pathlib import Path

import pytest
from PIL import Image

from dgx_monarch import demo_inputs

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "example_workflows").glob("*.json"))
SHIPPED = ROOT / "example_workflows" / "input"
COMFY_OWN = {"example.png"}  # a tracked file of every ComfyUI checkout
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov",
                  ".mp3", ".wav", ".flac", ".ogg")
NOTES = ("Note", "MarkdownNote")  # prose, where a link may end in a media suffix
# The frames a template takes from each clip: WanSCAILToVideo and
# WanAnimate2ToVideo read 81, FlowRVS's Image From Batch reads 17.
MIN_FRAMES = {"driving_video.mp4": 81, "driving_video_mask.mp4": 81,
              "wan_animate2_driving.mp4": 81, "source_video.mp4": 17}
# Where a graph states the frame size it works at: the width and height widgets
# of its Wan video node or of its Scale Image node, and the Qwen Image 2.1
# encoder's resolution, a square.
SIZE_WIDGETS = {"WanAnimate2ToVideo": (0, 1), "WanSCAILToVideo": (0, 1), "WanDancerVideo": (0, 1),
                "ImageScale": (1, 2), "TextEncodeQwenImage21": (2, 2)}
# Scale Image To Total Pixels at 1.0 megapixel, FLUX.3 Klein's reference scale,
# makes a square reference 1024 on a side.
MEGAPIXEL_SQUARE = (1024, 1024)


@pytest.fixture(autouse=True)
def _switch_unset(monkeypatch, _the_battery_places_no_demo_inputs):
    """Clear the battery's switch; return the value the battery had set."""
    battery = os.environ.get("DGXM_DEMO_INPUTS")
    monkeypatch.delenv("DGXM_DEMO_INPUTS", raising=False)
    return battery


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    logged: list[str] = []
    monkeypatch.setattr(demo_inputs.log, "warning", lambda message, *args: logged.append(message % args))
    return logged


def _nodes(graph: dict) -> list[dict]:
    found = list(graph.get("nodes", []))
    for subgraph in graph.get("definitions", {}).get("subgraphs", []):
        found.extend(_nodes(subgraph))
    return found


def _graphs() -> dict[str, list[dict]]:
    return {path.stem: _nodes(json.loads(path.read_text(encoding="utf-8"))) for path in TEMPLATES}


def _media(nodes: list[dict]) -> set[str]:
    """Every media file name a widget of any node but a note holds."""
    names: set[str] = set()
    for node in nodes:
        values = node.get("widgets_values")
        if node.get("type") in NOTES or not isinstance(values, list):
            continue
        names.update(value for value in values
                     if isinstance(value, str) and value.lower().endswith(MEDIA_SUFFIXES))
    return names


def _named() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for stem, nodes in _graphs().items():
        for name in _media(nodes):
            names.setdefault(name, set()).add(stem)
    return names


def _graph_sizes() -> dict[str, tuple[int, int]]:
    """Each demo image and clip -> the frame size the graphs that name it work at."""
    sizes: dict[str, set[tuple[int, int]]] = {}
    for nodes in _graphs().values():
        stated = set()
        for node in nodes:
            values = node.get("widgets_values") or []
            if node.get("type") in SIZE_WIDGETS:
                width, height = SIZE_WIDGETS[node["type"]]
                stated.add((values[width], values[height]))
            elif node.get("type") == "ImageScaleToTotalPixels" and values[1] == 1.0:
                stated.add(MEGAPIXEL_SQUARE)
        for name in _media(nodes) - COMFY_OWN:
            sizes.setdefault(name, set()).update(stated)
    assert all(len(stated) == 1 for stated in sizes.values() if stated), sizes
    return {name: next(iter(stated)) for name, stated in sizes.items() if stated}


def test_the_battery_turns_the_copy_off_for_every_other_test(_switch_unset):
    assert _switch_unset == "off"


def test_each_place_that_runs_the_entrypoint_turns_the_copy_off_itself():
    # Two places execute the pack entrypoint: the entrypoint canary, inside a
    # real ComfyUI tree, and the import-order probe, in a child process. Each
    # assigns the switch, so a shell that exports another value changes nothing.
    canary = (ROOT / "tests" / "canary" / "comfy_entrypoint_canary.py").read_text(encoding="utf-8")
    assert 'os.environ["DGXM_DEMO_INPUTS"] = "off"' in canary
    assert 'setdefault("DGXM_DEMO_INPUTS"' not in canary
    probe = (ROOT / "tests" / "test_monarch_diagnostics.py").read_text(encoding="utf-8")
    assert "DGXM_DEMO_INPUTS='off'" in probe


def test_every_demo_input_a_shipped_template_names_ships_with_the_pack():
    named = _named()
    assert len(named) > 20, "the scan found almost no media names; the widget layout moved"
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


def test_the_source_archive_carries_every_demo_input():
    # CI compares the sdist with the tracked files, and MANIFEST.in decides
    # what the sdist holds.
    patterns = [pattern for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
                if line.startswith(("recursive-include example_workflows ",
                                    "recursive-include example_workflows/input "))
                for pattern in line.split()[2:]]
    left_out = [path.name for path in demo_inputs.shipped()
                if not any(fnmatch(path.name, pattern) for pattern in patterns)]
    assert demo_inputs.shipped() and not left_out, f"MANIFEST.in leaves out {left_out}"


def test_every_demo_image_and_clip_has_a_size_its_graph_states():
    sized = _graph_sizes()
    unsized = {path.name for path in demo_inputs.shipped() if path.suffix in (".png", ".mp4")} - set(sized)
    assert not unsized, f"no graph states a frame size for {sorted(unsized)}"


@pytest.mark.parametrize("name", sorted(path.name for path in SHIPPED.glob("*.png")))
def test_each_demo_image_has_its_graphs_size_and_an_alpha_channel_where_one_is_read(name):
    with Image.open(SHIPPED / name) as image:
        image.load()
        wants_alpha = name.endswith("_rgba.png") or "alpha" in name
        assert image.mode == ("RGBA" if wants_alpha else "RGB")
        assert image.size == _graph_sizes()[name]
        if name in {"qwen21_mask_rgba.png", "qwen21_edit_mask_alpha.png"}:
            low, high = image.getchannel("A").getextrema()
            assert (low, high) == (0, 255), "the change guide needs a transparent region"


@pytest.mark.parametrize("name", sorted(MIN_FRAMES))
def test_each_demo_clip_decodes_at_its_graphs_size_with_the_frames_its_template_takes(name):
    av = pytest.importorskip("av", reason="PyAV, ComfyUI's video and audio reader, is absent here")
    with av.open(str(SHIPPED / name)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == _graph_sizes()[name]
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
    (tmp_path / "input").mkdir()
    fake = types.ModuleType("folder_paths")
    fake.get_input_directory = lambda: str(tmp_path / "input")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    assert demo_inputs.place_demo_inputs() == [path.name for path in demo_inputs.shipped()]


def _raises():
    raise RuntimeError("the surface moved")


@pytest.mark.parametrize("surface", [None, lambda: None, _raises],
                         ids=["no function", "returns None", "raises"])
def test_a_broken_comfy_surface_never_raises_into_startup(monkeypatch, warnings, surface):
    fake = types.ModuleType("folder_paths")
    if surface is not None:
        fake.get_input_directory = surface  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    assert demo_inputs.place_demo_inputs() == []
    assert len(warnings) == 1 and "in ComfyUI's input folder (" in warnings[0]


@pytest.mark.parametrize("value", ["off", "0", "false", "No"])
def test_the_switch_turns_placement_off(monkeypatch, tmp_path, value):
    monkeypatch.setenv("DGXM_DEMO_INPUTS", value)
    assert demo_inputs.place_demo_inputs(tmp_path) == []
    assert not any(tmp_path.iterdir())


def test_a_missing_input_folder_is_never_created(tmp_path, warnings):
    # A share not yet mounted must not gain a local folder the mount later hides.
    missing = tmp_path / "share" / "input"
    assert demo_inputs.place_demo_inputs(missing) == []
    assert not (tmp_path / "share").exists()
    assert len(warnings) == 1 and "never creates one" in warnings[0]


@pytest.mark.parametrize("case", ["a file in its place", "read-only"])
def test_a_folder_it_cannot_write_logs_a_warning_and_raises_nothing(tmp_path, warnings, case):
    blocked = tmp_path / "input"
    if case == "read-only":
        if os.geteuid() == 0:
            pytest.skip("root writes into a read-only folder")
        blocked.mkdir(mode=0o500)
    else:
        blocked.write_text("a file where the folder should be")
    assert demo_inputs.place_demo_inputs(blocked) == []
    assert len(warnings) == 1 and "supply your own" in warnings[0]
    assert blocked.is_file() or not any(blocked.iterdir())


def test_a_copy_that_fails_at_close_is_removed(monkeypatch, tmp_path, warnings):
    # Most of the files fit in the write buffer, so a full disk shows only
    # when the close flushes it.
    real_open = Path.open

    class FailingClose:
        def __init__(self, handle):
            self.handle = handle

        def write(self, data):
            return self.handle.write(data)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()
            raise OSError(28, "No space left on device")

    def open_failing(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        return FailingClose(handle) if mode == "xb" else handle

    monkeypatch.setattr(Path, "open", open_failing)
    assert demo_inputs.place_demo_inputs(tmp_path) == []
    assert not any(tmp_path.iterdir())
    assert len(warnings) == 1 and "No space left" in warnings[0]


def test_a_copy_that_stops_part_way_still_logs_what_it_placed(monkeypatch, tmp_path, warnings):
    names = [path.name for path in demo_inputs.shipped()]
    real_read = Path.read_bytes

    def read_failing(self):
        if self.name == names[2]:
            raise OSError(5, "Input/output error")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read_failing)
    assert demo_inputs.place_demo_inputs(tmp_path) == names[:2]
    assert sorted(path.name for path in tmp_path.iterdir()) == names[:2]
    assert len(warnings) == 1
    assert f"placed 2 template demo file(s) in {tmp_path} ({names[0]}, {names[1]}), then stopped" in warnings[0]


def test_a_link_in_the_shipped_folder_is_never_followed(monkeypatch, tmp_path):
    # ComfyUI serves its input folder over /view, so a link committed beside
    # the demo files must not carry another file of the box there.
    source, target, private = tmp_path / "shipped", tmp_path / "input", tmp_path / "private.png"
    source.mkdir()
    target.mkdir()
    private.write_bytes(b"not the pack's to copy")
    (source / "drawn.png").write_bytes(b"drawn")
    (source / "linked.png").symlink_to(private)
    monkeypatch.setattr(demo_inputs, "SOURCE", source)
    assert [path.name for path in demo_inputs.shipped()] == ["drawn.png"]
    assert demo_inputs.place_demo_inputs(target) == ["drawn.png"]
    assert [path.name for path in target.iterdir()] == ["drawn.png"]


def test_the_pack_entrypoint_places_the_demo_inputs_at_import():
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "place_demo_inputs"]
    assert len(calls) == 1 and not calls[0].args


def test_a_template_note_says_the_pack_places_its_demo_inputs_and_no_other_note_does():
    shipped = {path.name for path in demo_inputs.shipped()}
    for stem, nodes in _graphs().items():
        notes = " ".join(node["widgets_values"][0] for node in nodes if node.get("type") in NOTES)
        says_so = "The pack places" in notes and "DGXM_DEMO_INPUTS=off" in notes
        assert says_so == bool(_media(nodes) & shipped), stem


@pytest.mark.parametrize("page", ["docs/DESIGN.md", "docs/INSTALL.md", "docs/QUICKSTART.md"])
def test_the_docs_name_the_copy_and_its_switch(page):
    text = " ".join((ROOT / page).read_text(encoding="utf-8").split())
    assert "DGXM_DEMO_INPUTS=off" in text and "ComfyUI's input folder" in text
