#!/usr/bin/env python3
"""Write a LoRA that pins an attention kernel on a checkpoint's blocks, for a probe.

ComfyUI reads a uint8 JSON tensor at ``<block>.comfy_attention.config`` and
binds that block's attention to the kernel it names. A LoRA key of the same
shape, ``diffusion_model.<block>.comfy_attention.config``, stages the pin
through ComfyUI's own patcher (comfy/lora.py, ``ModelPatcher.add_patches``).
This tool reads only a checkpoint's safetensors header, finds the attention
modules of one family from their query projection keys, and writes a LoRA that
holds nothing but those config tensors, so a render with it exercises a real
pin without touching the checkpoint. docs/TROUBLESHOOTING.md #117 says how to
use it.

Key patterns, matched after ComfyUI's checkpoint prefix is stripped (the
parent of the query projection is the attention module):

  krea2         blocks.N.attn.wq.weight, txtfusion.{layerwise,refiner}_blocks.N.attn.wq.weight
  anima         blocks.N.{self,cross}_attn.q_proj.weight (the "net." prefix stripped;
                the LLM adapter's attention carries no pin and is skipped)
  kandinsky5    {text,visual}_transformer_blocks.N.{self,cross}_attention.to_query.weight
  wan21         blocks.N.{self,cross}_attn.q.weight
  qwen_image21  transformer_blocks.N.attn.to_q.weight
  flux3         {txt_mode_blocks,single_blocks,content_mode_blocks.{image,image_cond}}.N.{qkv,q_proj}.weight
                (the block owns its attention, so the module path is the block's;
                the int8 files fuse qkv, the bf16 and nvfp4 files split it)
  flux3_early   the same for the ten early blocks only (txt_mode_blocks and
                content_mode_blocks.image_cond), the stock sites comfy runs from
                preprocess_conds
  moss_image    {layers,text_embedder.blocks}.N.{proj.qkv,proj.q,qkv}.weight (each block owns
                its attention; fused blocks carry proj.qkv, split ones proj.q, text blocks qkv)
  moss_image_text  text_embedder.blocks.N.qkv.weight only, the two text blocks the
                sharded forward leaves replicated, so their pins run at stock sites

Only torch and safetensors are needed.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

# family -> pattern over the stripped key: group "path" is the attention module,
# and group "kind", where a family has one, tells self- from cross-attention.
FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "krea2": re.compile(
        r"^(?P<path>(?:txtfusion\.(?:layerwise|refiner)_blocks|blocks)\.\d+\.attn)\.wq\.weight$"),
    "anima": re.compile(r"^(?P<path>blocks\.\d+\.(?P<kind>self|cross)_attn)\.q_proj\.weight$"),
    "kandinsky5": re.compile(
        r"^(?P<path>(?:text|visual)_transformer_blocks\.\d+\.(?P<kind>self|cross)_attention)"
        r"\.to_query\.weight$"),
    "wan21": re.compile(r"^(?P<path>blocks\.\d+\.(?P<kind>self|cross)_attn)\.q\.weight$"),
    "qwen_image21": re.compile(r"^(?P<path>transformer_blocks\.\d+\.attn)\.to_q\.weight$"),
    "flux3": re.compile(
        r"^(?P<path>(?:txt_mode_blocks|single_blocks|content_mode_blocks\.(?:image|image_cond))"
        r"\.\d+)\.(?:qkv|q_proj)\.weight$"),
    "flux3_early": re.compile(
        r"^(?P<path>(?:txt_mode_blocks|content_mode_blocks\.image_cond)\.\d+)"
        r"\.(?:qkv|q_proj)\.weight$"),
    "moss_image": re.compile(
        r"^(?P<path>(?:layers|text_embedder\.blocks)\.\d+)\.(?:proj\.qkv|proj\.q|qkv)\.weight$"),
    "moss_image_text": re.compile(r"^(?P<path>text_embedder\.blocks\.\d+)\.qkv\.weight$"),
}
# ComfyUI's checkpoint prefixes (comfy/model_detection.py unet_prefix_from_state_dict).
PREFIXES = ("model.diffusion_model.", "model.model.", "net.")
KERNELS = ("comfy_kitchen_sol", "comfy_kitchen_int8")
MAX_HEADER_BYTES = 100 * 1024 * 1024


def read_header_keys(path: Path) -> list[str]:
    """Every tensor key in a safetensors file, read from its header alone."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if not 0 < length <= MAX_HEADER_BYTES:
            raise ValueError(f"{path} declares a {length}-byte header")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"{path} has no safetensors header")
    return [key for key in header if key != "__metadata__"]


def strip_prefix(keys: list[str]) -> list[str]:
    """Drop the checkpoint prefix ComfyUI strips, chosen by ComfyUI's own rule."""
    counts = {prefix: sum(key.startswith(prefix) for key in keys) for prefix in PREFIXES}
    top = max(counts, key=lambda prefix: counts[prefix])
    if counts[top] <= 5:
        return keys
    return [key[len(top):] for key in keys if key.startswith(top)]


def attention_paths(family: str, keys: list[str], *, cross: bool = False) -> list[str]:
    """The attention module paths of ``family`` in these keys, in block order.

    Self-attention only unless ``cross``: comfy-kitchen's Sol kernel takes q,
    k and v of one shape, which cross-attention does not give it.
    """
    try:
        pattern = FAMILY_PATTERNS[family]
    except KeyError:
        raise ValueError(f"unknown family {family!r}; known: {sorted(FAMILY_PATTERNS)}") from None
    paths = []
    for key in strip_prefix(keys):
        match = pattern.match(key)
        if match is None:
            continue
        kind = match.groupdict().get("kind")
        if kind == "cross" and not cross:
            continue
        paths.append(match.group("path"))

    def order(path: str) -> tuple:
        return tuple(int(part) if part.isdigit() else part for part in path.split("."))

    return sorted(set(paths), key=order)


def pin_config(kernel: str, tau: float | None) -> dict:
    """The JSON ComfyUI's ComfyAttention reads for one block."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; known: {', '.join(KERNELS)}")
    if kernel == "comfy_kitchen_sol":
        return {"attention": kernel, "tau": 1.0 if tau is None else float(tau)}
    if tau is not None:
        raise ValueError("--tau applies to comfy_kitchen_sol only")
    return {"attention": kernel}


def build_lora(paths: list[str], config: dict) -> dict:
    """LoRA tensors that pin ``config`` on every path, encoded as ComfyUI encodes it."""
    import torch

    encoded = list(json.dumps(config).encode("utf-8"))
    return {f"diffusion_model.{path}.comfy_attention.config":
            torch.tensor(encoded, dtype=torch.uint8) for path in paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("checkpoint", type=Path, help="the checkpoint's .safetensors file")
    parser.add_argument("out", type=Path, help="the LoRA file to write")
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_PATTERNS))
    parser.add_argument("--kernel", required=True, choices=KERNELS)
    parser.add_argument("--tau", type=float, default=None, help="Sol's tau (default 1.0)")
    parser.add_argument("--cross", action="store_true",
                        help="pin cross-attention too (int8 only: Sol needs self-attention)")
    parser.add_argument("--blocks", type=int, default=None,
                        help="pin only the first N attention modules")
    args = parser.parse_args(argv)
    if args.cross and args.kernel == "comfy_kitchen_sol":
        parser.error("comfy_kitchen_sol takes q, k and v of one shape; pin self-attention only")
    try:
        config = pin_config(args.kernel, args.tau)
        paths = attention_paths(args.family, read_header_keys(args.checkpoint), cross=args.cross)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.blocks is not None:
        paths = paths[:max(args.blocks, 0)]
    if not paths:
        print(f"error: no {args.family} attention keys matched "
              f"{FAMILY_PATTERNS[args.family].pattern} in {args.checkpoint.name}", file=sys.stderr)
        return 1
    if args.out.exists():
        print(f"error: {args.out} exists; pick another name", file=sys.stderr)
        return 2
    from safetensors.torch import save_file

    save_file(build_lora(paths, config), str(args.out),
              metadata={"dgx_monarch_pin_probe": json.dumps(config), "family": args.family})
    print(f"wrote {args.out}: {json.dumps(config)} on {len(paths)} attention modules "
          f"({paths[0]} .. {paths[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
