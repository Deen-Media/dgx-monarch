"""The recorded-schema builder keeps no file name a bundled workflow does not reference.

tests/fixtures/studio/build_object_info_bundled.py trims a recorded /object_info
before it becomes the committed fixture. A driver lists every file in its input
and model folders, private names included, and ComfyUI's image, audio and video
loaders accept more types than the builder's FILE_SUFFIX names. Any choice that
looks like a file, by a short extension or a slash, must go unless a workflow
references it. CPU only.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "dgxm_build_object_info_bundled", ROOT / "tests/fixtures/studio/build_object_info_bundled.py")
assert _SPEC and _SPEC.loader
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def test_planted_file_names_of_unlisted_types_do_not_enter_the_fixture():
    recording = {
        "LoadImage": {"input": {"required": {"image": [
            ["example.png", "planted_private_scan.tif", "planted_holiday.HEIC", "planted/sub/folder.png"],
            {"image_upload": True}]}}, "output": ["IMAGE", "MASK"]},
        "LoadAudio": {"input": {"required": {"audio": [
            ["reference_audio.mp3", "planted_voice_memo.m4a", "default"], {"audio_upload": True}]}},
            "output": ["AUDIO"]},
    }
    workflows = [{"workflow": {"nodes": [
        {"type": "LoadImage", "widgets_values": ["example.png", "image"]},
        {"type": "LoadAudio", "widgets_values": ["reference_audio.mp3"]},
    ]}}]
    fixture = builder.build(recording, workflows)
    assert fixture["LoadImage"]["input"]["required"]["image"][0] == ["example.png"]
    # A choice that is no file name, such as a mode, stays.
    assert fixture["LoadAudio"]["input"]["required"]["audio"][0] == ["reference_audio.mp3", "default"]
    assert "planted" not in repr(fixture)
