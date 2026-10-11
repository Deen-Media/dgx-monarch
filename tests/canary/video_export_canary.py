"""Execute the MiniMax template's native export tail with synthetic CPU AV.

No models, network, worker services or GPU work. COMFYUI_DIR selects the
ComfyUI checkout; the encoded file is temporary and removed after inspection.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.argv = ["dgxm-video-export-canary", "--cpu"]
    sys.path.insert(0, str(Path(os.environ.get("COMFYUI_DIR", "../ComfyUI")).resolve()))
    sys.path.insert(0, str(REPO))
    import comfy.options

    comfy.options.enable_args_parsing()
    import av
    import folder_paths
    import torch
    from benchmark.sweep.convert import convert, repo_widget_names
    from comfy_api.latest import _io
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    template = REPO / "example_workflows/dgx-monarch-minimax-h3-mp4.json"
    graph = convert(json.loads(template.read_text()), repo_widget_names())
    inputs = {node["class_type"]: dict(node["inputs"]) for node in graph.values()}
    create = inputs["CreateVideo"]
    create["images"] = torch.linspace(0, 1, 24 * 32 * 32 * 3).reshape(24, 32, 32, 3)
    tone = torch.sin(torch.arange(48000) * (2 * torch.pi * 440 / 48000)) * 0.1
    create["audio"] = {"waveform": tone.reshape(1, 1, -1), "sample_rate": 48000}
    video = CreateVideo.execute(**create)[0]
    save = inputs["SaveVideo"]
    save["video"] = video
    schema, _hidden, data = _io.get_finalized_class_inputs(SaveVideo.INPUT_TYPES(), save)
    assert set(schema["required"]) <= set(save), schema
    data["hidden_inputs"] = {}
    prepared = SaveVideo.PREPARE_CLASS_CLONE(data)
    old_output = folder_paths.get_output_directory()
    try:
        with tempfile.TemporaryDirectory(prefix="dgxm-video-export-") as output:
            folder_paths.set_output_directory(output)
            prepared.execute(**_io.build_nested_inputs(save, data))
            paths = list(Path(output).rglob("*.mp4"))
            assert len(paths) == 1, paths
            with av.open(str(paths[0])) as container:
                vstream, = container.streams.video
                astream, = container.streams.audio
                assert vstream.codec_context.name == "h264"
                assert astream.codec_context.name == "aac"
                assert vstream.average_rate == 24
                assert len(list(container.decode(video=0))) == 24
            with av.open(str(paths[0])) as container:
                audio = list(container.decode(audio=0))
                assert audio and sum(frame.samples for frame in audio) >= 48000
            print("native export green: MP4, H.264, AAC, 24 fps, 24 frames; synthetic CPU AV")
    finally:
        folder_paths.set_output_directory(old_output)


if __name__ == "__main__":
    main()
