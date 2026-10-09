#!/usr/bin/env python3
"""Count the T5 tokens a Chroma prompt sends to the model.

Ulysses pads odd text lengths and excludes those pad rows from attention
(since 2026-09-03). Ring and hybrid lack that full-sequence exclusion point;
padding requires the ring_pad waiver. Shipped prompts remain even.

ComfyUI's PixArtTokenizer uses no start token, no maximum-length padding,
and a minimum length of one. For plain text, its count is the tokenizer's
length including one end token. Weighted text and embeddings are tokenized
in separate segments; this script refuses them rather than reporting an
incorrect count.

Requires ``tokenizers`` and ComfyUI's ``comfy/text_encoders/t5_tokenizer/
tokenizer.json``. Set --tokenizer or COMFYUI_DIR (default ~/ComfyUI).

    python tools/count_t5_tokens.py                    # Chroma workflows
    python tools/count_t5_tokens.py example_workflows/dgx-monarch-chroma-t2i.json
    python tools/count_t5_tokens.py --text "a prompt"
    python tools/count_t5_tokens.py --require-even     # exit 1 on odd counts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The graphs whose prompts must count even. Radiance is in the set because
# ChromaRadiance subclasses Chroma and runs the same adapter
# (src/dgx_monarch/adapters/flux_family.py), so its text stream takes the same
# pad and the same row exclusion.
CHROMA_TEMPLATES = (
    REPO / "example_workflows" / "dgx-monarch-chroma-t2i.json",
    REPO / "example_workflows" / "dgx-monarch-radiance-t2i.json",
    REPO / "tests" / "fixtures" / "workflows" / "generated" / "dgx-monarch-test-chroma-even.json",
    REPO / "tests" / "fixtures" / "workflows" / "generated" / "dgx-monarch-test-chroma-lora.json",
)

TOKENIZER_SUFFIX = Path("comfy") / "text_encoders" / "t5_tokenizer" / "tokenizer.json"

# Comfy segments a prompt at weight parentheses and at `embedding:`; brackets
# split nothing in comfy, and the counter refuses them anyway.
_SEGMENTING = ("(", ")", "[", "]", "embedding:")

# The shipped positive before 2026-09-02, kept as a fixed anchor: the uly2
# divisibility refusal of that prompt named 101 tokens, so a tokenizer that
# counts it otherwise is not the one comfy loads.
CALIBRATION_TEXT = (
    "This is a nature documentary close-up photograph of the right side of "
    "the face of a tiger. The photograph is centered on it's highly detailed "
    "and speckled eye surrounded by intricately detailed fur. Overlaid at the "
    "center of the image is a title text that says \"CHROMA1-HD\" in a large "
    "white 3D letters. Amateur photography. Unfiltered. Real life. Natural "
    "light. Subtle shadows. "
)
CALIBRATION_COUNT = 101


def tokenizer_path(explicit: str | None = None) -> Path:
    """Where the T5 tokenizer file is, by argument, env var, or convention."""
    if explicit:
        return Path(explicit).expanduser()
    comfy = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))
    return Path(comfy).expanduser() / TOKENIZER_SUFFIX


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def count_tokens(text: str, tokenizer) -> int:
    """The stream length comfy hands chroma for ``text``.

    ``tokenizer`` is anything with the ``tokenizers`` encode contract: an
    ``encode(text)`` returning an object whose ``ids`` is the token list with
    the end token on it.
    """
    for marker in _SEGMENTING:
        if marker in text:
            raise ValueError(
                f"prompt carries {marker!r}; this counter reads only prompts with no parentheses, "
                "brackets or embedding:name references, since comfy tokenizes such a prompt as one segment"
            )
    return len(tokenizer.encode(text).ids)


def calibrate(tokenizer) -> None:
    """Refuse to report counts from a tokenizer that misses the known one."""
    got = count_tokens(CALIBRATION_TEXT, tokenizer)
    if got != CALIBRATION_COUNT:
        raise RuntimeError(
            f"tokenizer counts the calibration prompt at {got}, not "
            f"{CALIBRATION_COUNT}; it is not the one comfy loads for chroma"
        )


def prompts_in(path: Path) -> list[tuple[str, str]]:
    """Every text a CLIPTextEncode carries in a UI-format workflow."""
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    found = []
    for node in document["nodes"]:
        if node.get("type") != "CLIPTextEncode":
            continue
        values = node.get("widgets_values") or []
        if values and isinstance(values[0], str):
            found.append((str(node.get("id")), values[0]))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workflow", nargs="*",
                        help="UI-format workflow JSON (default: the chroma set, unless --text is given)")
    parser.add_argument("--text", action="append", default=[],
                        help="count this text (repeatable); it replaces the default chroma set")
    parser.add_argument("--tokenizer", help="path to the T5 tokenizer.json (default: under COMFYUI_DIR or ~/ComfyUI)")
    parser.add_argument("--require-even", action="store_true",
                        help="exit 1 when any prompt counts odd")
    args = parser.parse_args(argv)

    path = tokenizer_path(args.tokenizer)
    if not path.is_file():
        print(f"no tokenizer at {path}; pass --tokenizer or set COMFYUI_DIR",
              file=sys.stderr)
        return 2
    try:
        tokenizer = load_tokenizer(path)
    except ImportError:
        print("the tokenizers package is not installed", file=sys.stderr)
        return 2
    try:
        calibrate(tokenizer)
    except RuntimeError as bad:
        print(bad, file=sys.stderr)
        return 2

    files = [Path(name) for name in args.workflow]
    if not files and not args.text:
        files = list(CHROMA_TEMPLATES)

    odd = 0
    try:
        for text in args.text:
            count = count_tokens(text, tokenizer)
            odd += count % 2
            print(f"{count:>5}  {'even' if count % 2 == 0 else 'ODD '}  {text[:60]!r}")
        for workflow in files:
            print(workflow.name)
            for node_id, text in prompts_in(workflow):
                count = count_tokens(text, tokenizer)
                odd += count % 2
                print(f"  node {node_id:>3}  {count:>5}  "
                      f"{'even' if count % 2 == 0 else 'ODD '}  {text[:50]!r}")
    except ValueError as split:
        print(split, file=sys.stderr)
        return 2
    if odd and args.require_even:
        print(f"{odd} prompt(s) count odd", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
