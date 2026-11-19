"""Synthetic MiniMax-H3-Image DiT headers, with no weights.

The key names, dtypes, shapes, uint8 marker payloads and ``config`` metadata of
the three pruned DiT files Comfy-Org/MiniMax-H3-Image publishes (BF16, INT8
ConvRot and W4A8), read from their headers on 2026-11-10. ``write`` lays a
header down and leaves the data a hole as long as the real file's, with only
the small uint8 markers written, so a 40 GB file costs a few kilobytes on disk.
``blocks`` shrinks the block count for tests that need no byte shares.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

GRID = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724,
        0.23200887441635132, 0.0]
CONFIG = {"transformer": {"pdd_steps": 6, "pdd_sigmas": GRID}}
METADATA = {"config": json.dumps(CONFIG)}
PIN = json.dumps([{"attention": "comfy_kitchen_int8"}])
QUANT = {
    "int8_convrot": json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}),
    "w4a8": json.dumps({"format": "asym_w4a8_int8"}),
    "nvfp4": json.dumps({"format": "nvfp4"}),
}
KINDS = ("bf16", "int8_convrot", "w4a8")
WIDTH = 5376
# The four block linears, (rows, input columns).
LINEARS = {"attn.qkv_proj": (21504, 5376), "attn.out_proj": (5376, 7168),
           "mlp.fc1": (28672, 5376), "mlp.fc2": (5376, 14336)}
BASIS = 8
ADALN_ROWS = {"blocks": 96768, "final_layer": 10752}
WIDTHS = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I8": 1, "F8_E4M3": 1}


def _linear(name: str, rows: int, cols: int, kind: str) -> dict:
    if kind == "bf16":
        return {f"{name}.weight": ("BF16", [rows, cols])}
    marker = QUANT[kind].encode()
    if kind == "int8_convrot":
        return {f"{name}.weight": ("I8", [rows, cols]),
                f"{name}.weight_scale": ("F32", [rows, 1]),
                f"{name}.comfy_quant": ("U8", [len(marker)], marker)}
    if kind == "w4a8":
        return {f"{name}.weight": ("I8", [rows, cols // 2]),
                f"{name}.weight_codebook": ("F32", [16]),
                f"{name}.weight_s_channel": ("F32", [rows]),
                f"{name}.weight_s_rel": ("F8_E4M3", [rows, cols // 16]),
                f"{name}.comfy_quant": ("U8", [len(marker)], marker)}
    # A community 4-bit export in ComfyUI's nvfp4 layout: no published file, no row.
    return {f"{name}.weight": ("U8", [rows, cols // 2]),
            f"{name}.weight_scale": ("F8_E4M3", [rows, cols // 16]),
            f"{name}.weight_scale_2": ("F32", []),
            f"{name}.comfy_quant": ("U8", [len(marker)], marker)}


def tensors(kind: str = "bf16", *, blocks: int = 50, pins: bool = True,
            audio: bool = False) -> dict[str, tuple]:
    """``{name: (dtype, shape)}``, or ``(dtype, shape, payload)`` for a uint8 marker."""
    out: dict[str, tuple] = {
        "adaln_t_table": ("F32", [1025, BASIS]),
        "condition_proj.weight": ("BF16", [WIDTH, 5120]),
        "condition_proj.bias": ("BF16", [WIDTH]),
        "final_layer.adaln_proj.linear.weight": ("F16", [ADALN_ROWS["final_layer"], BASIS]),
        "final_layer.adaln_proj.linear.bias": ("BF16", [ADALN_ROWS["final_layer"]]),
        "final_layer.norm.weight": ("BF16", [WIDTH]),
        "final_layer.video_out.weight": ("F32", [576, WIDTH]),
        "final_layer.video_out.bias": ("F32", [576]),
        "rope.inv_freq": ("F32", [16]),
        "token_refiner.final_norm.weight": ("BF16", [WIDTH]),
        "video_patch_proj.weight": ("F32", [WIDTH, 96]),
        "video_patch_proj.bias": ("F32", [WIDTH]),
    }
    if audio:  # MiniMax H3's audio path: ComfyUI tells the two apart by its patch projection
        out["audio_patch_proj.weight"] = ("BF16", [WIDTH, 128])
        out["final_layer.audio_out.weight"] = ("F32", [96, WIDTH])
    for index in range(2):
        prefix = f"token_refiner.blocks.{index}"
        for name, (rows, cols) in LINEARS.items():
            out[f"{prefix}.{name}.weight"] = ("BF16", [rows, cols])
        for name, size in (("attn.q_norm", 128), ("attn.k_norm", 128), ("norm1", WIDTH),
                           ("norm2", WIDTH)):
            out[f"{prefix}.{name}.weight"] = ("BF16", [size])
    for index in range(blocks):
        prefix = f"blocks.{index}"
        out[f"{prefix}.adaln_proj.linear.weight"] = ("F16", [ADALN_ROWS["blocks"], BASIS])
        out[f"{prefix}.adaln_proj.linear.bias"] = ("BF16", [ADALN_ROWS["blocks"]])
        if pins:
            out[f"{prefix}.attn.comfy_attention.config"] = ("U8", [len(PIN)], PIN.encode())
        for name, (rows, cols) in LINEARS.items():
            out.update({f"{prefix}.{key}": value
                        for key, value in _linear(name, rows, cols, kind).items()})
        for name, size in (("attn.q_norm", 128), ("attn.k_norm", 128), ("norm1", WIDTH),
                           ("norm2", WIDTH)):
            out[f"{prefix}.{name}.weight"] = ("BF16", [size])
    return out


def _nbytes(dtype: str, shape: list[int]) -> int:
    count = WIDTHS[dtype]
    for dim in shape:
        count *= dim
    return count


def write(path: Path, kind: str = "bf16", *, metadata: dict | None = METADATA,
          extra: dict[str, tuple] | None = None, drop: tuple[str, ...] = (), **options) -> Path:
    """Write one sparse safetensors file; ``extra`` adds tensors, ``drop`` removes them."""
    table = {**tensors(kind, **options), **(extra or {})}
    header: dict = {"__metadata__": dict(metadata)} if metadata is not None else {}
    payloads = []
    offset = 0
    for name, (dtype, shape, *payload) in table.items():
        if name in drop:
            continue
        size = _nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        if payload:
            payloads.append((offset, payload[0]))
        offset += size
    blob = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        base = 8 + len(blob)
        for start, data in payloads:
            handle.seek(base + start)
            handle.write(data)
        handle.truncate(base + offset)
    return path
