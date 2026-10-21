"""Build object_info_bundled.json from a recorded driver /object_info response.

Usage::

    python tests/fixtures/studio/build_object_info_bundled.py RECORDING.json

The fixture keeps only the node classes the bundled Studio workflows use, keeps
only the fields Studio reads, drops tooltips and descriptions, and cuts every
file-list choice down to the names those workflows reference. A list left with
no name gets one neutral placeholder. Choices that are not file names, such as
sampler names, stay whole. Every referenced name must be in the recording,
except the example names in NOT_IN_RECORDING, which the recording rig had not
installed; the build adds those and stops on any other missing name.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / "src/dgx_monarch/studio/data/workflows.json"
OUTPUT = Path(__file__).with_name("object_info_bundled.json")
FILE_SUFFIX = re.compile(r"\.(safetensors|gguf|sft|ckpt|pt|pth|bin|onnx|png|jpe?g|webp|gif|mp4|webm|mov|mkv"
                         r"|wav|flac|mp3|ogg|opus|json|yaml|txt|glb)$", re.IGNORECASE)
KEEP = ("input", "input_order", "output", "output_is_list", "output_name", "output_node", "name", "python_module")
VIRTUAL = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}
# Example names that public bundled workflows ship and the recording rig lacked.
NOT_IN_RECORDING = {
    "DGXMonarchLoraLoader": {"your_first_lora.safetensors", "your_second_lora.safetensors"},
    "LoadImage": {"blue_dancer.png", "qwen21_canvas_rgba.png", "qwen21_mask_rgba.png", "qwen21_source.png",
                  "reference_character.png", "reference_character_mask.png"},
    "LoadVideo": {"driving_video.mp4", "driving_video_mask.mp4", "source_video.mp4"},
    "LoadAudio": {"wan_dancer_reference_audio.mp3"},
}


def _nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    found = list(graph.get("nodes", []))
    for subgraph in graph.get("definitions", {}).get("subgraphs", []):
        found.extend(_nodes(subgraph))
    return found


def referenced(workflows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Return, per node class, every string widget value the bundled workflows set."""
    names: dict[str, set[str]] = {}
    for row in workflows:
        for node in _nodes(row["workflow"]):
            values = node.get("widgets_values") or []
            values = values.values() if isinstance(values, dict) else values
            names.setdefault(node["type"], set()).update(value for value in values if isinstance(value, str))
    return names


def _file_like(choice: Any) -> bool:
    return isinstance(choice, str) and (FILE_SUFFIX.search(choice) is not None or "/" in choice)


def _trim_choices(choices: list[Any], names: set[str], added: set[str]) -> list[Any]:
    if not any(_file_like(choice) for choice in choices):
        return choices
    kept = [choice for choice in choices if not _file_like(choice) or choice in names]
    kept.extend(sorted(added - set(kept)))
    if not any(_file_like(choice) for choice in kept):
        suffix = next(FILE_SUFFIX.search(choice).group(0) for choice in choices
                      if isinstance(choice, str) and FILE_SUFFIX.search(choice))
        kept.append("placeholder" + suffix)
    return kept


def _file_lists(spec: Any) -> list[list[Any]]:
    """Every choice list in one input spec, including dynamic option inputs."""
    lists = []
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(spec[0], list):
        lists.append(spec[0])
    if spec[0] == "COMBO" and isinstance(options.get("options"), list):
        lists.append(options["options"])
    for row in options.get("options", []) if spec[0] == "COMFY_DYNAMICCOMBO_V3" else []:
        for group in (row.get("inputs") or {}).values():
            for child in (group or {}).values():
                lists.extend(_file_lists(child))
    return [choices for choices in lists if any(_file_like(choice) for choice in choices)]


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key not in {"tooltip", "description"}}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def build(recording: dict[str, Any], workflows: list[dict[str, Any]]) -> dict[str, Any]:
    refs = referenced(workflows)
    missing_classes = sorted(kind for kind in refs if kind not in VIRTUAL and kind not in recording)
    if missing_classes:
        raise SystemExit(f"recording lacks node classes: {missing_classes}")
    fixture = {}
    for kind in sorted(set(refs) - VIRTUAL):
        row = copy.deepcopy({key: recording[kind][key] for key in KEEP if key in recording[kind]})
        names = refs[kind]
        lists = [choices for group in ("required", "optional")
                 for spec in row["input"].get(group, {}).values() for choices in _file_lists(spec)]
        recorded = {choice for choices in lists for choice in choices}
        wanted = {name for name in names if FILE_SUFFIX.search(name)}
        absent = wanted - recorded
        if absent - NOT_IN_RECORDING.get(kind, set()):
            raise SystemExit(f"{kind}: referenced names missing from the recording: {sorted(absent)}")
        if NOT_IN_RECORDING.get(kind, set()) & recorded:
            raise SystemExit(f"{kind}: NOT_IN_RECORDING names are in the recording; remove them from the list")
        if absent and len(lists) != 1:
            raise SystemExit(f"{kind}: cannot tell which file list takes {sorted(absent)}")
        for choices in lists:
            choices[:] = _trim_choices(choices, names, absent)
        fixture[kind] = _strip(row)
        for group in ("required", "optional"):
            for spec in fixture[kind]["input"].get(group, {}).values():
                options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                pool = spec[0] if isinstance(spec[0], list) else options.get("options")
                if isinstance(pool, list) and "default" in options and options["default"] not in pool:
                    raise SystemExit(f"{kind}: a default left its trimmed choice list")
    return fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path, help="a saved /object_info JSON response")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    workflows = json.loads(WORKFLOWS.read_text())["workflows"]
    fixture = build(json.loads(args.recording.read_text()), workflows)
    args.out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(fixture)} node classes to {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
