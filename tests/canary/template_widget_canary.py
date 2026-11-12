#!/usr/bin/env python3
"""Validate committed templates against ComfyUI master's node schemas on CPU.

Signature canaries do not cover template widgets. A widget added, removed or
retyped upstream can invalidate a converted graph or shift its values. Against
ComfyUI's nodes.NODE_CLASS_MAPPINGS and this pack's nodes, this canary checks:

1. Widget names and order match INPUT_TYPES, excluding frontend-added widgets,
   and each dynamic combo key the generator lists is one ComfyUI offers and
   reveals the same child widgets in the same order (dynamic_widget_errors).
2. Every required input has a link or widget value.
3. Every supplied input remains declared by the node.
4. CLIPLoader type values belong to ComfyUI's choices.
5. CreateVideo bit_depth, color_space, chroma_subsampling and codec remain
   optional, with values allowed by create_video_enum_errors, and SaveVideo's
   format and codec stay among ComfyUI's options (save_video_errors).
6. No widget that ComfyUI marks dynamicPrompts holds text the browser would
   rewrite on Queue (dynamic_prompt_errors). The API path keeps that text, so
   such a graph would render one prompt from the browser and another from the
   API, the sweep and Studio.
7. The MiniMax-H3-Image caption read gives the caption its table states for
   each answer, through ComfyUI's own Extract Text, Replace Text (Regex),
   Trim Text and Compare Text nodes (parse_chain_errors).

Conversion uses benchmark.sweep.convert.convert and object_info_widget_names,
the same functions used by the live sweep.

The templates follow master. A stock widget that an older ComfyUI this
project supports lacks is a row in dgx_monarch/stock_widget_drift.py: on a
ComfyUI without it, the order check puts it back at its place, and the
conversion drops it only where the template holds its default. The canary
cannot tell an older ComfyUI from a master that later removed the widget
(master reports the last release's version), so a removal reads as the older
case here and the table row needs a look when upstream touches that node
again.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Values a template may use: offered by ComfyUI 0.43 and by master. Master's
# prores and mov are left out because 0.43 lacks them.
CREATE_VIDEO_CHOICES = {"bit_depth": ("auto", 8, 10),
                        "color_space": ("sRGB", "HDR", "HDR PQ"),
                        "chroma_subsampling": ("auto", "444", "422", "420"),
                        "codec": ("none", "auto", "h264", "av1")}
SAVE_VIDEO_FORMATS = ("auto", "mp4", "mkv", "webm")
SAVE_VIDEO_CODECS = ("auto", "h264", "av1")
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


def _choices(spec: object) -> list | None:
    """A combo's offered values, from either declaration shape, else None."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return list(spec[0])
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    values = options.get("options")
    if isinstance(values, list):
        return [row.get("key") if isinstance(row, dict) else row for row in values]
    return None


def create_video_enum_errors(schema: dict, inputs: dict) -> list[str]:
    """Validate the static CreateVideo combos before the queue sees a graph.

    A value must be one this repo allows and one this ComfyUI offers. A newer
    widget this ComfyUI lacks reaches here only at a value other than its
    default, which the template check has already reported, so it also reads
    as having no choices.
    """
    errors = []
    for name, choices in CREATE_VIDEO_CHOICES.items():
        if name not in inputs:
            continue
        try:
            offered = _choices(schema["optional"][name])
        except (KeyError, IndexError, TypeError):
            errors.append(f"CreateVideo has no {name} choices")
            continue
        if inputs[name] not in choices:
            errors.append(f"CreateVideo {name} {inputs[name]!r} is not one of {list(choices)!r}")
        elif offered is not None and inputs[name] not in offered:
            errors.append(f"CreateVideo {name} {inputs[name]!r} is not offered here: {offered!r}")
    return errors


def save_video_errors(schema: dict, inputs: dict) -> list[str]:
    """SaveVideo's format, and the codec under it, stay among ComfyUI's options."""
    errors = []
    fmt, codec = inputs.get("format"), inputs.get("codec")
    try:
        formats = schema["required"]["format"][1]["options"]
    except (KeyError, IndexError, TypeError):
        return ["SaveVideo has no format options"]
    if fmt is None:
        return errors
    row = next((row for row in formats if isinstance(row, dict) and row.get("key") == fmt), None)
    if fmt not in SAVE_VIDEO_FORMATS or row is None:
        return [f"SaveVideo format {fmt!r} is not one of {list(SAVE_VIDEO_FORMATS)!r} offered here"]
    if codec is not None:
        nested = _choices(((row.get("inputs") or {}).get("required") or {}).get("codec"))
        if codec not in SAVE_VIDEO_CODECS or nested is None or codec not in nested:
            errors.append(f"SaveVideo codec {codec!r} is not offered under format {fmt!r}: {nested!r}")
    return errors


# The frontend's dynamic prompt pass, ported from processDynamicPrompt in
# comfyui_frontend_package 1.57.6 (formatUtil). Its Comfy.DynamicPrompts
# extension runs it on Queue over every widget whose input spec sets
# dynamicPrompts to true, and the API prompt takes the result. Comments go
# first; then every unescaped `{` opens a choice block that ends at its
# matching `}`, and one `|`-separated option replaces the block, braces and
# all, recursively. A block with no `|` keeps its text but loses its braces, so
# a JSON object loses every brace pair. `\{`, `\}` and `\|` lose their
# backslash. JavaScript's `.` stops at these four line ends, and reading past
# the end of the text gives "undefined", which the escape keeps.
_DYNAMIC_PROMPT_COMMENT = re.compile(r"/\*[\s\S]*?\*/|//[^\n\r\u2028\u2029]*")
_DYNAMIC_PROMPT_ESCAPE = re.compile(r"\\([{}|])")


def process_dynamic_prompt(text: str,
                           choose: Callable[[list[str]], str] = random.choice) -> str:
    """What the browser sends for one dynamicPrompts widget holding ``text``."""
    text = _DYNAMIC_PROMPT_COMMENT.sub("", text)
    pos = 0

    def escape() -> str:
        nonlocal pos
        char = text[pos] if pos < len(text) else "undefined"
        pos += 1
        return "\\" + char

    def choice_block() -> str:
        nonlocal pos
        options: list[str] = []
        current, depth = "", 0
        while pos < len(text):
            char = text[pos]
            pos += 1
            if char == "\\":
                current += escape()
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                if not depth:
                    break
                depth -= 1
            elif char == "|" and not depth:
                options.append(current)
                current = ""
                continue
            current += char
        options.append(current)
        return process_dynamic_prompt(choose(options), choose)

    out: list[str] = []
    while pos < len(text):
        char = text[pos]
        pos += 1
        out.append(escape() if char == "\\" else choice_block() if char == "{" else char)
    return _DYNAMIC_PROMPT_ESCAPE.sub(r"\1", "".join(out))


def dynamic_prompt_widgets(schema: dict) -> list[str]:
    """The inputs this schema marks for the browser's dynamic prompt pass."""
    names = []
    for section in ("required", "optional"):
        for name, spec in (schema.get(section) or {}).items():
            if (isinstance(spec, (list, tuple)) and len(spec) > 1
                    and isinstance(spec[1], dict) and spec[1].get("dynamicPrompts") is True):
                names.append(name)
    return names


def dynamic_prompt_errors(schema: dict, inputs: dict) -> list[str]:
    """A dynamicPrompts widget value the browser would send changed.

    A wired input arrives as a link, never as its widget's text, so only a
    string value counts. An unescaped `{`, a comment, an escaped brace or bar,
    or a trailing backslash changes the text whichever option the pass picks,
    and nothing else changes it, so one pass decides it.
    """
    errors = []
    for name in dynamic_prompt_widgets(schema):
        value = inputs.get(name)
        if isinstance(value, str) and process_dynamic_prompt(value) != value:
            errors.append(
                f"{name} holds text the browser's dynamic prompt pass rewrites on Queue "
                f"({value[:40]!r}); feed it from Text (Multiline) through a link")
    return errors


def dynamic_widget_errors(node_class: str, generated: list, declared_names: list,
                          schema: dict) -> list[str]:
    """The generator's dynamic combo rows against this ComfyUI's schema.

    tools/gen_templates.py DYNAMIC_WIDGETS lists, per dynamic combo, the keys
    a committed graph may hold and the child widgets each reveals. Every
    listed key must be one ComfyUI offers, and must reveal the same widgets
    in the same order: the converter reads a graph's ``widgets_values`` by
    them, so a child ComfyUI adds, drops or moves shifts every later value.
    """
    errors = []
    for name in generated:
        listed = getattr(name, "children", None)
        if listed is None:
            continue
        spec = (schema.get("required") or {}).get(name) or (schema.get("optional") or {}).get(name)
        offered = _choices(spec) or []
        declared = next((other for other in declared_names if other == name), None)
        revealed = getattr(declared, "children", {})
        for key, children in listed.items():
            if key not in offered:
                errors.append(f"widget map: {node_class}.{name} lists the key {key!r}, "
                              f"which comfy master does not offer: {offered!r}")
            elif [str(child) for child in children] != [str(c) for c in revealed.get(key, [])]:
                errors.append(
                    f"widget map: {node_class}.{name} key {key!r} generated children "
                    f"{[str(child) for child in children]} but comfy master reveals "
                    f"{[str(c) for c in revealed.get(key, [])]}")
    return errors


# ---- The MiniMax-H3-Image caption read ------------------------------------------
#
# The shipped MiniMax-H3-Image graphs read the expander's caption by pattern:
# Extract Text takes the value of the answer's outermost wrapper, Replace Text
# (Regex) turns a backslash before a double quote, a backslash or a slash into
# that mark, and Trim Text and Compare Text decide whether a caption was read.
# The table below is the contract: each answer, the caption the image model
# gets, and "read" or "request" (no caption, so the graph sends the request as
# written). tests/test_minimax_h3_image_templates.py runs it through stand-ins
# for the four nodes; main() runs it through ComfyUI's own.
PARSE_TEMPLATE = "dgx-monarch-minimax-h3-image-t2i.json"
PARSE_NODES = ("RegexExtract", "RegexReplace", "StringTrim", "StringCompare")
PARSE_CASES = (
    ("a valid object",
     '{"expanded_caption": "A photograph of a boat."}',
     "A photograph of a boat.", "read"),
    ("escaped double quotes",
     '{"expanded_caption": "A sign that reads \\"OPEN\\" on a door."}',
     'A sign that reads "OPEN" on a door.', "read"),
    ("an escaped double quote at the caption's end",
     '{"expanded_caption": "The sign says \\"OK\\""}',
     'The sign says "OK"', "read"),
    ("an escaped backslash",
     '{"expanded_caption": "Chalk text \'A\\\\B = C\' on a board."}',
     "Chalk text 'A\\B = C' on a board.", "read"),
    ("an escaped path",
     '{"expanded_caption": "A label that reads C:\\\\new\\\\temp."}',
     "A label that reads C:\\new\\temp.", "read"),
    ("an escaped slash",
     '{"expanded_caption": "A sign that says 1\\/2 off."}',
     "A sign that says 1/2 off.", "read"),
    ("a code fence",
     '```json\n{"expanded_caption": "A photograph of a boat."}\n```',
     "A photograph of a boat.", "read"),
    ("text before and after the object",
     'Here is the caption: {"expanded_caption": "A photograph of a boat."} I hope it helps.',
     "A photograph of a boat.", "read"),
    ("inner double quotes",
     '{"expanded_caption": "A card that says "status ok" in black letters."}',
     'A card that says "status ok" in black letters.', "read"),
    ("an inner JSON object copied as typed",
     '{"expanded_caption": "A card with the text {"status": "ok"} in black. Brackets float beside it."}',
     'A card with the text {"status": "ok"} in black. Brackets float beside it.', "read"),
    ("an inner object that itself holds the answer key",
     '{"expanded_caption": "A poster that reads {"expanded_caption": "hello"} in red."}',
     'A poster that reads {"expanded_caption": "hello"} in red.', "read"),
    ("a raw double quote then a closing brace inside the caption",
     '{"expanded_caption": "A sign with the marks "} painted on it, in a blue room."}',
     'A sign with the marks "} painted on it, in a blue room.', "read"),
    ("a raw double quote at the very end of the caption",
     '{"expanded_caption": "The sign says "OK""}',
     'The sign says "OK"', "read"),
    ("raw and escaped double quotes in one caption",
     '{"expanded_caption": "A \\"big\\" dog by a sign that says "OPEN"."}',
     'A "big" dog by a sign that says "OPEN".', "read"),
    ("inner double quotes inside a code fence, with text after",
     '```json\n{"expanded_caption": "A "big" dog."}\n```\nDone.',
     'A "big" dog.', "read"),
    ("a raw path whose backslashes look like JSON escapes stays as typed",
     '{"expanded_caption": "A label that reads C:\\new\\temp."}',
     "A label that reads C:\\new\\temp.", "read"),
    ("a raw path with other backslashes",
     '{"expanded_caption": "A label that reads C:\\data\\files."}',
     "A label that reads C:\\data\\files.", "read"),
    ("backslash-escaped quotes around the value",
     '{\n"expanded_caption": \\"A yellow circle on a wall.\\"}"',
     "A yellow circle on a wall.", "read"),
    ("an escaped and a plain closing quote",
     '{\n"expanded_caption": \\"A black bicycle.\\""\n}',
     "A black bicycle.", "read"),
    ("a raw double quote then a comma inside the caption",
     '{"expanded_caption": "A sign reads ", then more."}',
     'A sign reads ", then more.', "read"),
    ("an inner object with two keys at the caption's end",
     '{"expanded_caption": "A card with the text {"status": "ok", "code": "7"}"}',
     'A card with the text {"status": "ok", "code": "7"}', "read"),
    ("a second key after the caption",
     '{"expanded_caption": "A cat on a mat.", "aspect_ratio": "1:1"}',
     "A cat on a mat.", "read"),
    ("more keys on their own lines, a number and true among them",
     '{\n  "expanded_caption": "A cat on a mat.",\n  "seed": 7,\n  "safe": true\n}',
     "A cat on a mat.", "read"),
    ("a second key whose string holds an escaped quote",
     '{"expanded_caption": "A cat on a mat.", "note": "say \\"hi\\""}',
     "A cat on a mat.", "read"),
    ("inner double quotes, then a second key",
     '{"expanded_caption": "A "big" cat.", "aspect_ratio": "1:1"}',
     'A "big" cat.', "read"),
    ("a key before the caption",
     '{"aspect_ratio": "1:1", "expanded_caption": "A cat on a mat."}',
     "A cat on a mat.", "read"),
    ("a planning object before the answer",
     '{"plan": "one cat, one mat"} {"expanded_caption": "A cat on a mat."}',
     "A cat on a mat.", "read"),
    # The limits of reading by pattern, kept here so a change to any is seen.
    ("limit: an escaped line break stays as typed",
     '{"expanded_caption": "Line one.\\nLine two."}',
     "Line one.\\nLine two.", "read"),
    ("limit: an escaped tab stays as typed",
     '{"expanded_caption": "Left.\\tRight."}',
     "Left.\\tRight.", "read"),
    ("limit: a unicode escape stays as typed",
     '{"expanded_caption": "A caf\\u00e9 sign."}',
     "A caf\\u00e9 sign.", "read"),
    ("limit: two objects are read as one caption",
     '{"expanded_caption": "The first."} {"expanded_caption": "The second."}',
     'The first."} {"expanded_caption": "The second.', "read"),
    ("limit: a first object that is not valid joins the second",
     '{"expanded_caption": "A} {"expanded_caption": "C"}',
     'A} {"expanded_caption": "C', "read"),
    ("limit: a note after the object that holds a quote then a brace joins the caption",
     '{"expanded_caption": "A cat."} Note: the form is {"expanded_caption": "..."}',
     'A cat."} Note: the form is {"expanded_caption": "...', "read"),
    ("limit: a second key with an object as its value joins the caption",
     '{"expanded_caption": "A cat.", "meta": {"a": "b"}}',
     'A cat.", "meta": {"a": "b', "read"),
    ("limit: a drawn backslash written unescaped before a double quote is read as an escape",
     '{"expanded_caption": "A sign that reads \'x\\"y\' on a wall."}',
     'A sign that reads \'x"y\' on a wall.', "read"),
    ("limit: a drawn backslash written unescaped before a slash is read as an escape",
     '{"expanded_caption": "A face \'\\/\' drawn on a wall."}',
     "A face '/' drawn on a wall.", "read"),
    ("a raw line break inside the caption is kept",
     '{"expanded_caption": "Line one.\nLine two."}',
     "Line one.\nLine two.", "read"),
    ("limit: a second key with a list as its value loses the caption",
     '{"expanded_caption": "A cat.", "tags": ["a", "b"]}', "", "request"),
    ("an object as the value", '{"expanded_caption": {"text": "A cat."}}', "", "request"),
    ("null as the value", '{"expanded_caption": null}', "", "request"),
    ("a trailing comma after the caption", '{"expanded_caption": "A cat.",}', "", "request"),
    ("an answer cut short", '{"expanded_caption": "A photograph of a small weather sta', "", "request"),
    ("an empty value", '{"expanded_caption": ""}', "", "request"),
    ("a blank value", '{"expanded_caption": "   "}', "", "request"),
    ("plain prose", "A photograph of a boat on a beach.", "", "request"),
    ("another key", '{"caption": "A photograph of a boat."}', "", "request"),
    ("an empty answer", "", "", "request"),
)


def parse_settings(api: dict) -> dict:
    """The caption read's widget values, from a converted MiniMax-H3-Image graph."""
    def only(kind: str) -> dict:
        (inputs,) = [node["inputs"] for node in api.values() if node["class_type"] == kind]
        return inputs

    read, marks = only("RegexExtract"), only("RegexReplace")
    return {"read": tuple(read[name] for name in (
                "regex_pattern", "mode", "case_insensitive", "multiline", "dotall", "group_index")),
            "marks": tuple(marks[name] for name in (
                "regex_pattern", "replace", "case_insensitive", "multiline", "dotall", "count"))}


def caption_from(answer: str, settings: dict, run: Callable) -> tuple[str, str]:
    """The caption the graph's read gives for one answer, or ("", "request").

    ``run(node_class, *args)`` executes one node and returns its first output.
    """
    caption = run("RegexReplace", run("RegexExtract", answer, *settings["read"]),
                  *settings["marks"])
    if run("StringCompare", run("StringTrim", caption, "Both"), "", "Equal", True):
        return "", "request"
    return caption, "read"


def parse_chain_errors(settings: dict, run: Callable) -> list[str]:
    errors = []
    for what, answer, caption, read in PARSE_CASES:
        got = caption_from(answer, settings, run)
        if got != (caption, read):
            errors.append(f"caption read: {what}: expected {(caption, read)!r}, got {got!r}")
    return errors


def _parse_chain_failures(node_classes: dict, names: Callable) -> list[str]:
    """The table through ComfyUI's own nodes, with the shipped graph's widget values."""
    if any(kind not in node_classes for kind in PARSE_NODES):
        return []  # an older ComfyUI: the template's own check says what it lacks
    from benchmark.sweep import convert as sweep_convert

    try:
        api = sweep_convert.convert(
            json.loads((TEMPLATE_DIRS[0] / PARSE_TEMPLATE).read_text(encoding="utf-8")), names)
    except (KeyError, ValueError):
        return []

    def run(kind: str, *args):
        return node_classes[kind].execute(*args).result[0]

    return parse_chain_errors(parse_settings(api), run)


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

    from dgx_monarch import stock_widget_drift

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
        # A newer widget this ComfyUI lacks goes back at its place first, so
        # an older ComfyUI still checks every other widget's slot
        # (stock_widget_drift).
        declared_order = stock_widget_drift.with_newer(node_class, comfy_names[node_class])
        if generated != declared_order:
            failures.append(
                f"widget map: {node_class} generated as {generated} but comfy "
                f"master declares {comfy_names[node_class]}"
            )
        failures.extend(dynamic_widget_errors(node_class, generated, comfy_names[node_class],
                                              declared))

    # The MiniMax-H3-Image caption read, through ComfyUI's own four nodes.
    failures.extend(_parse_chain_failures(node_classes, names))

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

    from dgx_monarch import stock_widget_drift

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
        newer = stock_widget_drift.missing(node_class, names(node_class))
        for name in sorted(set(node["inputs"]) - valid):
            if name in newer:
                failures.append(
                    f"{label}: node {node_id} {node_class} holds "
                    f"{node['inputs'][name]!r} for {name}, which this ComfyUI lacks; "
                    f"a template must hold {newer[name]!r} there"
                )
                continue
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
        if node_class == "SaveVideo":
            for error in save_video_errors(declared, node["inputs"]):
                failures.append(f"{label}: node {node_id} {error}")
        for error in dynamic_prompt_errors(schemas[node_class], node["inputs"]):
            failures.append(f"{label}: node {node_id} {node_class} {error}")
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
