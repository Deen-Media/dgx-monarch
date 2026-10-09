#!/usr/bin/env python3
"""Validate committed templates against ComfyUI master's node schemas on CPU.

Signature canaries do not cover template widgets. A widget added, removed or
retyped upstream can invalidate a converted graph or shift its values. Against
ComfyUI's nodes.NODE_CLASS_MAPPINGS and this pack's nodes, this canary checks:

1. Widget names and order match INPUT_TYPES, excluding frontend-added widgets.
2. Every required input has a link or widget value.
3. Every supplied input remains declared by the node.
4. CLIPLoader type values belong to ComfyUI's choices.
5. CreateVideo bit_depth, color_space and codec remain optional, with values
   allowed by create_video_enum_errors.

Conversion uses benchmark.sweep.convert.convert and object_info_widget_names,
the same functions used by the live sweep.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", "../ComfyUI")).expanduser().resolve()
TEMPLATE_DIRS = (
    REPO / "example_workflows",
    REPO / "tests" / "fixtures" / "workflows" / "generated",
)
# Widgets the frontend adds beside a declared input, which comfy therefore
# never declares itself. Each holds a slot in widgets_values, so the generator
# emits it and this canary has to allow it on the generator's side alone.
# Where each comes from, in comfyui_frontend_package 1.53.6:
#   control_after_generate: appended to a seed INT (see gen_templates)
#   upload: added by Comfy.UploadImage to a node with a required combo whose
#           options carry image_upload, video_upload or animated_image_upload,
#           and by Comfy.UploadAudio when the `audio` input carries audio_upload
#   audioUI: the player Comfy.AudioWidget adds to LoadAudio and the audio save
#            and preview nodes
FRONTEND_WIDGETS = frozenset({"control_after_generate", "upload", "audioUI"})


def clip_loader_type_error(schema: dict, value: object) -> str | None:
    """Return the schema error for one serialized CLIPLoader type value."""
    try:
        choices = schema["required"]["type"][0]
    except (KeyError, IndexError, TypeError):
        return "CLIPLoader schema has no type choices"
    if value not in choices:
        return f"CLIPLoader type {value!r} is not one of {list(choices)!r}"
    return None


def create_video_enum_errors(schema: dict, inputs: dict) -> list[str]:
    """Validate the static CreateVideo combos before the queue sees a graph."""
    errors = []
    pinned = {"bit_depth": ("auto", 8, 10),
              "color_space": ("sRGB", "HDR", "HDR PQ"),
              "codec": ("none", "auto", "h264", "av1")}
    for name, choices in pinned.items():
        if name not in inputs:
            continue
        try:
            schema["optional"][name]
        except (KeyError, IndexError, TypeError):
            errors.append(f"CreateVideo has no {name} choices")
            continue
        if inputs[name] not in choices:
            errors.append(f"CreateVideo {name} {inputs[name]!r} is not one of {list(choices)!r}")
    return errors


def main() -> None:
    node_classes = _comfy_and_pack_nodes()
    from comfy_api.internal import _ComfyNodeInternal
    from comfy_api.latest import _io

    def finalized_inputs(node_class: str, declared: dict, inputs: dict) -> dict:
        # Match Comfy's V3 prompt validation: autogrow groups expand into their
        # actual named sockets and an empty min=0 group is not a missing input.
        if issubclass(node_classes[node_class], _ComfyNodeInternal):
            return _io.get_finalized_class_inputs(declared, inputs)[0]
        return declared
    templates = _templates()
    assert templates, f"no templates found under {TEMPLATE_DIRS[0]}"

    from benchmark.sweep import convert as sweep_convert

    generator_names = sweep_convert.repo_widget_names(REPO)
    schemas: dict[str, dict] = {}
    comfy_names: dict[str, list[str]] = {}

    def names(node_class: str) -> list[str]:
        if node_class not in comfy_names:
            cls = node_classes.get(node_class)
            if cls is None:
                raise KeyError(node_class)
            schemas[node_class] = cls.INPUT_TYPES()
            comfy_names[node_class] = sweep_convert.object_info_widget_names(
                schemas[node_class]
            )
        return comfy_names[node_class]

    failures: list[str] = []
    placed: set[str] = set()
    for path in templates:
        placed |= _check_template(path, names, schemas, failures, finalized_inputs)

    # Order check last: it needs every class the templates place, and reporting
    # the per-template symptom first puts the render-facing failure on top.
    for node_class in sorted(placed):
        try:
            generated = generator_names(node_class)
        except KeyError:
            failures.append(
                f"widget map: the generator does not know {node_class}, "
                "which a template places"
            )
            continue
        declared = schemas[node_class]
        for name in sorted(FRONTEND_WIDGETS & set(generated)):
            if name in (declared.get("required") or {}) or name in (
                    declared.get("optional") or {}):
                failures.append(
                    f"widget map: {node_class} declares an input named {name}, "
                    "which this canary treats as frontend-added"
                )
        generated = [name for name in generated if name not in FRONTEND_WIDGETS]
        if generated != comfy_names[node_class]:
            failures.append(
                f"widget map: {node_class} generated as {generated} but comfy "
                f"master declares {comfy_names[node_class]}"
            )

    if failures:
        for line in failures:
            print(f"FAIL {line}")
        raise SystemExit(
            f"{len(failures)} template widget failures against comfy at {COMFY_DIR}"
        )
    print(
        f"template widgets green: {len(templates)} templates, "
        f"{len(placed)} node classes, comfy={COMFY_DIR}"
    )


def _check_template(path: Path, names, schemas: dict[str, dict],
                    failures: list[str], finalized_inputs=None) -> set[str]:
    """Convert one template, report what comfy would refuse, and return the classes it places."""
    from benchmark.sweep import convert as sweep_convert

    ui = json.loads(path.read_text())
    label = path.relative_to(REPO)
    placed: set[str] = set()
    try:
        api = sweep_convert.convert(ui, names)
    except KeyError as exc:
        failures.append(f"{label}: comfy master has no node {exc}")
        return placed

    for node_id, node in sorted(api.items(), key=lambda item: int(item[0])):
        node_class = node["class_type"]
        placed.add(node_class)
        declared = schemas[node_class]
        if finalized_inputs is not None:
            declared = finalized_inputs(node_class, declared, node["inputs"])
        required = list(declared.get("required") or {})
        valid = set(required) | set(declared.get("optional") or {})
        for name in required:
            if name not in node["inputs"]:
                failures.append(
                    f"{label}: node {node_id} {node_class} leaves required input "
                    f"{name} unset"
                )
        for name in sorted(set(node["inputs"]) - valid):
            failures.append(
                f"{label}: node {node_id} {node_class} sets {name}, which comfy "
                "master no longer declares"
            )
        if node_class == "CLIPLoader" and "type" in node["inputs"]:
            error = clip_loader_type_error(declared, node["inputs"]["type"])
            if error:
                failures.append(f"{label}: node {node_id} {error}")
        if node_class == "CreateVideo":
            for error in create_video_enum_errors(declared, node["inputs"]):
                failures.append(f"{label}: node {node_id} {error}")
    return placed


def _templates() -> list[Path]:
    return [path for directory in TEMPLATE_DIRS if directory.is_dir()
            for path in sorted(directory.glob("*.json"))]


def _comfy_and_pack_nodes() -> dict:
    """Comfy master's node classes on CPU, plus this pack's own."""
    sys.argv = ["dgxm-template-widget-canary", "--cpu"]
    sys.path.insert(0, str(COMFY_DIR))

    # Comfy ignores --cpu unless parsing is enabled before its device modules
    # load, the way main.py does it for a real server process.
    import comfy.options

    comfy.options.enable_args_parsing()

    # Install an unstarted PromptServer with a real route table, the same
    # stand-in comfy_entrypoint_canary.py builds and checks. The pack import
    # below (through gen_templates) registers its routes on it; with no
    # instance, nodes/routes.py register() returns early. ComfyUI's own node
    # loads do not need the stand-in (comfy a7169322, 2026-09-29):
    # nodes_replacements.py fails its on_load either way, and load_custom_node
    # logs that and goes on.
    import server
    from aiohttp import web

    if getattr(server.PromptServer, "instance", None) is None:
        prompt_server = object.__new__(server.PromptServer)
        prompt_server.routes = web.RouteTableDef()
        prompt_server.on_prompt_handlers = []
        server.PromptServer.instance = prompt_server

    import nodes

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))

    # Repo imports come after comfy. tools/gen_templates.py keeps its comfy
    # stubs out of sys.modules except around its own widget reads, and puts
    # the real modules back after each, so the real tree loaded above stays
    # the one this script compares against.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))
    import gen_templates

    return dict(nodes.NODE_CLASS_MAPPINGS) | dict(gen_templates.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    main()
