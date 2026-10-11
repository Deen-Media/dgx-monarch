"""The MP4 variant changes export only and preserves joint AV sampling."""
import json
from pathlib import Path

from benchmark.sweep.convert import convert, repo_widget_names

ROOT = Path(__file__).resolve().parents[1]


def _graph(name):
    return convert(json.loads((ROOT / "example_workflows" / name).read_text()),
                   repo_widget_names())


def test_mp4_variant_preserves_every_generation_input_and_both_decoders():
    base = _graph("dgx-monarch-minimax-h3-t2va.json")
    mp4 = _graph("dgx-monarch-minimax-h3-mp4.json")
    generation = {key: node for key, node in base.items()
                  if node["class_type"] not in {"SaveImage", "SaveAudioAdvanced"}}
    assert {key: node for key, node in mp4.items()
            if node["class_type"] not in {"CreateVideo", "SaveVideo"}} == generation
    assert not any(node["class_type"] in {"SaveImage", "SaveAudioAdvanced"}
                   for node in mp4.values())
    by_type = {node["class_type"]: (key, node["inputs"]) for key, node in mp4.items()}
    video_id, video = by_type["CreateVideo"]
    assert video == {"images": [by_type["VAEDecode"][0], 0],
                     "audio": [by_type["VAEDecodeAudio"][0], 0],
                     "fps": 24.0, "bit_depth": 8, "color_space": "sRGB", "codec": "none"}
    assert by_type["SaveVideo"][1] == {
        "video": [video_id, 0], "filename_prefix": "video/dgx_monarch_minimax_h3",
        "format": "mp4", "codec": "h264"}
